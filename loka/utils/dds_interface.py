"""DDS wrapper around unitree_sdk2_python LowCmd / LowState / SportModeState."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

NUM_MOTORS = 29


class DDSInterface:
    """Thin publisher/subscriber façade for G1 low-level and sport-mode topics.

    Parameters
    ----------
    domain_id:
        DDS domain. Simulation typically uses ``1``; real robot uses ``0``.
    interface:
        Network interface name (``\"lo\"`` for sim loopback).
    """

    def __init__(
        self,
        domain_id: int = 1,
        interface: str = "lo",
        *,
        topic_lowcmd: str = "rt/lowcmd",
        topic_lowstate: str = "rt/lowstate",
        topic_sportmodestate: str = "rt/sportmodestate",
        initialize_factory: bool = True,
    ) -> None:
        self.domain_id = domain_id
        self.interface = interface
        self._ready = False

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is required. Install unitree_sdk2_python into the "
                "active environment (see README)."
            ) from exc

        try:
            if initialize_factory:
                ChannelFactoryInitialize(domain_id, interface)
        except Exception as exc:
            raise RuntimeError(
                f"DDS ChannelFactoryInitialize({domain_id}, '{interface}') failed: {exc}"
            ) from exc

        self._LowCmd_ = LowCmd_
        self._LowState_ = LowState_
        self._SportModeState_ = SportModeState_
        self.crc = CRC()

        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd.mode_pr = 0
        self.cmd.mode_machine = 0
        for i in range(NUM_MOTORS):
            self.cmd.motor_cmd[i].mode = 1
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd_pub = ChannelPublisher(topic_lowcmd, LowCmd_)
        self.cmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber(topic_lowstate, LowState_)
        self.sport_sub = ChannelSubscriber(topic_sportmodestate, SportModeState_)

        self._lowstate_cb: Optional[Callable] = None
        self._sport_cb: Optional[Callable] = None
        self._ready = True
        logger.info(
            "DDSInterface ready (domain=%s, iface=%s)", domain_id, interface
        )

    @property
    def ready(self) -> bool:
        return self._ready

    def subscribe_lowstate(self, handler: Callable, queue_len: int = 10) -> None:
        """Subscribe to ``rt/lowstate`` (unitree_hg)."""
        self._lowstate_cb = handler
        self.lowstate_sub.Init(handler, queue_len)

    def subscribe_sportmodestate(self, handler: Callable, queue_len: int = 10) -> None:
        """Subscribe to ``rt/sportmodestate`` (unitree_go SportModeState_)."""
        self._sport_cb = handler
        self.sport_sub.Init(handler, queue_len)

    def publish_lowcmd(
        self,
        *,
        mode: Sequence[int],
        q: Sequence[float],
        dq: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        tau: Sequence[float],
    ) -> None:
        """Fill and publish a full 29-motor LowCmd_ payload with CRC."""
        if not (
            len(mode)
            == len(q)
            == len(dq)
            == len(kp)
            == len(kd)
            == len(tau)
            == NUM_MOTORS
        ):
            raise ValueError(f"all motor arrays must have length {NUM_MOTORS}")

        for i in range(NUM_MOTORS):
            self.cmd.motor_cmd[i].mode = int(mode[i])
            self.cmd.motor_cmd[i].q = float(q[i])
            self.cmd.motor_cmd[i].dq = float(dq[i])
            self.cmd.motor_cmd[i].kp = float(kp[i])
            self.cmd.motor_cmd[i].kd = float(kd[i])
            self.cmd.motor_cmd[i].tau = float(tau[i])

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.cmd_pub.Write(self.cmd)

    def publish_zero(self, motor_mode: int = 0x0A) -> None:
        """Emergency zero-torque hold on all motors."""
        zeros = np.zeros(NUM_MOTORS, dtype=np.float64)
        modes = np.full(NUM_MOTORS, motor_mode, dtype=np.int32)
        self.publish_lowcmd(
            mode=modes, q=zeros, dq=zeros, kp=zeros, kd=zeros, tau=zeros
        )
