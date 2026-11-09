import os, sys
from deploy.g1_spec import G1_SPEC

def test_button_bits_match_the_teleop_keymap():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "g1_walk"))
    import paths; paths.add_teleop()
    from teleop.robot_control.remote_controller import KeyMap
    r = G1_SPEC.remote
    assert (r.button_start, r.button_stop, r.button_reset, r.button_rearm_modifier) == (KeyMap.start, KeyMap.X, KeyMap.Y, KeyMap.L1)

def test_plot_offsets_come_from_g1_command():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "policy_rollout"))
    import g1_command
    assert tuple(G1_SPEC.plot.eef_offset) == tuple(g1_command.EEF_POSITION_OFFSET)
    assert G1_SPEC.plot.torso_height_offset == g1_command.TORSO_HEIGHT_OFFSET

def test_standing_defaults():
    s = G1_SPEC.standing
    assert (s.stand_height, s.height_tol, s.tilt_max, s.gyro_max, s.leg_dq_max, s.lowstate_max_age, s.hold_s) == (0.78, 0.06, 0.10, 0.25, 0.5, 0.1, 2.0)
