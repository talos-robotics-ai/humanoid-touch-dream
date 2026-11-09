"""The operator side of the walker link, built from the teleop's own classes.

Under ``--lower jetson`` the teleop's ``G1Backend`` builds, in ``_setup_policy``
and ``_setup_lower``:

* ``G1_29_JetsonBody(connect_link())`` -- the G1 body controller with no
  rt/lowcmd publisher: it decodes rt/lowstate, reads the Unitree remote (A
  freezes, L2+B damps) and forwards every arm command over the link;
* ``build_remote_transport`` / ``build_remote_lower`` -- the stand-ins for the
  transport and the walking composer that run in the walker process.

:class:`WalkerLink` builds exactly those, and runs the backend's handover
(``_start_policy``), session engage (``_engage_lower``) and fault read
(``_lower_fault``) by calling the backend's own methods on itself: each of
them reads only ``self._lower`` and ``self._tx``, which this class carries
under those names. Nothing here publishes rt/lowcmd or rt/arm_sdk: the body
class is built with ``PUBLISHES = False`` and has no publisher to write with.

Import after ``paths.add_teleop()``.
"""
from teleop.robot_control.g1_backend import _NO_HEAD, G1Backend
from teleop.robot_control.g1_jetson_remote import (G1_29_JetsonBody, build_remote_lower,
                                                   build_remote_transport, connect_link)
from teleop.utils.logger import logger


class WalkerLink:
    """The body, the transport and the composer stand-ins, connected to a
    running ``g1_jetson_walker.py``."""

    def __init__(self, host, port, estop_hook):
        self.link = connect_link(host, port)
        self.body = G1_29_JetsonBody(self.link)
        # Hooked the moment the body exists, as G1Backend._setup_policy does:
        # its state thread is already reading the remote's A button.
        self.body.estop_hook = estop_hook
        self._tx = build_remote_transport(self.body)
        self._lower = build_remote_lower(self._tx)
        self.strategy = self._lower.strategy

    # -- the backend's own lifecycle steps -----------------------------------
    def handover(self):
        """Start the composer and hand the legs to the walking policy (gain
        ramp, glide, parity check, blend), as ``G1Backend._start_policy``
        does; True once the policy is live."""
        timeout = G1Backend._handover_timeout(self._lower)
        self._lower.start()
        logger.info("[g1_walk] handing the legs to the walking policy (up to %.0f s)"
                    % timeout)
        handle = self._lower.prepare(timeout=timeout)
        if not handle.completed:
            logger.critical("[g1_walk] the walking-policy handover was refused: %s"
                            % (handle.reason or self._lower.trip_reason))
        return bool(handle.completed)

    def engage(self):
        """Whether the walking policy is driving the legs at a session start
        (``G1Backend._engage_lower``)."""
        return G1Backend._engage_lower(self)

    def fault(self):
        """``lower_<reason>`` once the composer or its transport latched,
        else None (``G1Backend._lower_fault``)."""
        return G1Backend._lower_fault(self)

    @property
    def live(self):
        return self.strategy.phase == "live"

    def applied_torso(self):
        s = self.strategy
        return (s.height_command, s.roll_command, s.pitch_command, s.yaw_command)

    # -- per tick ------------------------------------------------------------
    def send_arms(self, q, tau):
        """The arm command and the walker's arm reference, as the backend
        sends them after its jump guard (``_safely_set_arms`` then
        ``set_upper_targets``)."""
        self.body.ctrl_arms(q, tau)
        self._lower.set_upper_targets(q, _NO_HEAD)

    def detach(self):
        """Leave the walker running: close the link without the composer's
        stop. The walker finds the link silent after its stale bound, brings
        the walk to zero, holds the arms and keeps balancing."""
        logger.info("[g1_walk] detaching from the walker: it keeps the robot "
                    "standing; stop it in its own terminal")
        self.body.disarm_link_watchdog()
        self.link.close()
        self.body.stop_threads()
