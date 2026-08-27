#!/usr/bin/env python3
"""Unified G1 stand + walk entry (alias for ``loka.run_stand_loka``).

    python -m loka.run_loka
    python -m loka.run_loka --mode walk --speed 0.3 --heading 0.2
    python -m loka.run_loka --operator "walk forward at 0.25 m/s"
"""

from loka.run_stand_loka import main

if __name__ == "__main__":
    raise SystemExit(main())
