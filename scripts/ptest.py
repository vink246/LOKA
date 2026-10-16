#!/usr/bin/env python3
"""Run pytest node ids in parallel shards, without pytest-xdist.

    python scripts/ptest.py tests/test_gait.py tests/test_walk_transitions.py
    python scripts/ptest.py tests -j 20 -- -m "not slow"

Arguments before ``--`` are what pytest should collect; arguments after it
are passed to every shard. Tests are dealt round-robin into ``-j`` shards
(default: CPU count - 2), each run in its own pytest process. The closed-loop
tests are single-threaded MuJoCo, so wall time drops roughly with the shard
count. Prints each shard's failures and a combined summary; exits non-zero if
any shard failed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|error|errors)")


def _collect(targets: list[str], extra: list[str]) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *extra, *targets],
        capture_output=True,
        text=True,
    )
    ids = [line.strip() for line in out.stdout.splitlines() if "::" in line]
    if not ids:
        sys.stderr.write(out.stdout + out.stderr)
    return ids


def main() -> int:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", default=["tests"])
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)

    ids = _collect(args.targets, passthrough)
    if not ids:
        print("no tests collected")
        return 1
    jobs = max(1, min(args.jobs, len(ids)))
    shards = [ids[i::jobs] for i in range(jobs)]
    print(f"{len(ids)} tests in {jobs} shards")

    start = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    procs = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, shard in enumerate(shards):
            ids_file = Path(tmp) / f"shard{index}.txt"
            ids_file.write_text("\n".join(shard))
            log = open(Path(tmp) / f"shard{index}.log", "w")
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", *passthrough, *shard]
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        totals: dict[str, int] = {}
        failures: list[str] = []
        status = 0
        for index, (proc, log) in enumerate(procs):
            proc.wait()
            log.close()
            text = (Path(tmp) / f"shard{index}.log").read_text()
            if proc.returncode not in (0, 5):
                status = 1
            failures += [line for line in text.splitlines() if line.startswith(("FAILED ", "ERROR "))]
            tail = text.strip().splitlines()[-1] if text.strip() else ""
            for count, word in _SUMMARY.findall(tail):
                word = "errors" if word == "error" else word
                totals[word] = totals.get(word, 0) + int(count)
    for line in sorted(failures):
        print(line)
    summary = ", ".join(f"{n} {k}" for k, n in totals.items())
    print(f"{summary} in {time.perf_counter() - start:.0f}s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
