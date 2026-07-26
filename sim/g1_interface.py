#!/usr/bin/env python3
"""DDS low-level interface for Unitree G1 (unitree_hg).

Talks to either unitree_mujoco (domain 1, lo) or a real G1 (domain 0, NIC).
Future LOKA telemetry can read motor_state[i].tau_est from self.state.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Optional, Sequence

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

NUM_MOTORS = 29
DEFAULT_SIM_DOMAIN_ID = 1
DEFAULT_SIM_INTERFACE = "lo"
DEFAULT_REAL_DOMAIN_ID = 0


class LOKA_G1_Interface:
    """Publish LowCmd_ and subscribe LowState_ for the G1 (29 DoF)."""

    def __init__(
        self,
        domain_id: int = DEFAULT_SIM_DOMAIN_ID,
        interface: str = DEFAULT_SIM_INTERFACE,
    ):
        ChannelFactoryInitialize(domain_id, interface)

        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.state_handler, 10)

        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.state = unitree_hg_msg_dds__LowState_()
        self.crc = CRC()

        self.cmd.mode_pr = 0
        self.cmd.mode_machine = 0
        for i in range(NUM_MOTORS):
            self.cmd.motor_cmd[i].mode = 1  # enable
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

    def state_handler(self, msg: LowState_) -> None:
        self.state = msg
        # LOKA telemetry can monitor self.state.motor_state[i].tau_est here

    def send_torques(self, torques: Sequence[float]) -> None:
        """Map WBC / controller torques onto the 29 G1 motors (torque-only)."""
        if len(torques) != NUM_MOTORS:
            raise ValueError(f"expected {NUM_MOTORS} torques, got {len(torques)}")

        for i in range(NUM_MOTORS):
            self.cmd.motor_cmd[i].mode = 1
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = float(torques[i])

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.cmd_pub.Write(self.cmd)

    def send_zeros(self) -> None:
        self.send_torques([0.0] * NUM_MOTORS)


def init_from_argv(argv: Optional[Iterable[str]] = None) -> LOKA_G1_Interface:
    """Sim: no args → domain 1 / lo. Real: pass network interface name."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return LOKA_G1_Interface(DEFAULT_SIM_DOMAIN_ID, DEFAULT_SIM_INTERFACE)
    return LOKA_G1_Interface(DEFAULT_REAL_DOMAIN_ID, args[0])


if __name__ == "__main__":
    g1 = init_from_argv()
    print("LOKA_G1_Interface ready; sending zero torques at 500 Hz (Ctrl+C to stop)")
    while True:
        g1.send_zeros()
        time.sleep(0.002)
