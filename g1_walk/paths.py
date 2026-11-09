"""Where the two code bases this module joins live, and how both become
importable in one process.

The teleop's modules import each other as ``teleop.<...>`` from its repo root
and resolve their assets from their own ``__file__``; a few (the arm spec
loader, ``armkit``) are imported by bare name, so the teleop's package
directory goes on ``sys.path`` too, as its scripts put it there -- but LAST,
so its top-level ``utils`` never shadows HTD's module of the same name. HTD's
robot_control modules import each other by bare name and load their assets
from paths relative to ``robot_control/`` (``./assets/...``), which is why HTD
runs its controller from that directory and :func:`enter_htd_robot_control`
does the same.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HTD_ROOT = os.path.dirname(HERE)
HTD_ROBOT_CONTROL = os.path.join(HTD_ROOT, "robot_control")
TELEOP_REPO = os.path.join(HERE, "teleop-cbf-perception")
TELEOP_PKG = os.path.join(TELEOP_REPO, "teleop")
# The G1 arm spec the pinned teleop runs: its [pose] home is the stand pose
# the arms park at between sessions.
G1_ARM_SPEC = os.path.join(TELEOP_REPO, "assets", "g1", "arm_spec.toml")


def add_teleop():
    """The pinned teleop importable as ``teleop.*``."""
    if not os.path.isfile(os.path.join(TELEOP_PKG, "robot_control", "g1_jetson_remote.py")):
        raise RuntimeError("the pinned teleop is not checked out at %s: run "
                           "bash g1_walk/setup.sh" % TELEOP_REPO)
    if TELEOP_REPO not in sys.path:
        sys.path.insert(0, TELEOP_REPO)
    if TELEOP_PKG not in sys.path:
        sys.path.append(TELEOP_PKG)


def enter_htd_robot_control():
    """HTD's robot_control importable by bare name, with the working
    directory its relative asset paths assume."""
    if HTD_ROBOT_CONTROL not in sys.path:
        sys.path.insert(0, HTD_ROBOT_CONTROL)
    os.chdir(HTD_ROBOT_CONTROL)
