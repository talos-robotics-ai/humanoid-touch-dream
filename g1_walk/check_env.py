#!/usr/bin/env python3
"""Import everything the walker and the HTD-side controller load, without
opening DDS, Modbus or the link, and report the versions that matter.

    python g1_walk/check_env.py

Exit status 1 on the first import that fails.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

MODULES = (
    # the walker's network and the pinned teleop's client and hand driver
    "torch",
    "teleop.robot_control.g1_jetson_remote",
    "teleop.robot_control.robot_hand_inspire_modbus",
    # HTD robot_control, by bare name from robot_control/
    "robot_arm_ik_g1",
    "robot_state_estimator_g1",
    "robot_hand_inspire_dftp",
    "teleop_dds_msgs",
    # the touch message type policy_rollout subscribes with
    "inspire_sdkpy",
)


def main():
    paths.add_teleop()
    paths.enter_htd_robot_control()
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            print("FAILED: import %s: %r" % (name, e), file=sys.stderr)
            return 1
    import numpy
    import pinocchio
    import pymodbus
    import torch
    cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
    print("ok: imports; numpy %s, pinocchio %s, pymodbus %s, torch %s on %s"
          % (numpy.__version__, pinocchio.__version__, pymodbus.__version__,
             torch.__version__, cuda))
    from power import power_problems
    for problem in power_problems():
        print("WARNING: not ready for real-time control: %s" % problem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
