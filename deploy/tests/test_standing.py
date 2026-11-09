from deploy.g1_spec import G1_SPEC
from deploy.standing import StandingCheck, StandingSample

GOOD = dict(walker_live=True, torso_height=0.78, roll=0.0, pitch=0.02, gyro_norm=0.05, leg_dq_max=0.1, lowstate_age_s=0.02)

def test_needs_hold_time():
    c = StandingCheck(G1_SPEC.standing)
    r = c.update(StandingSample(**GOOD), 0.0)
    assert (r.standing, r.failing) == (False, ()) and r.held_s == 0.0
    r = c.update(StandingSample(**GOOD), 1.9)
    assert not r.standing and abs(r.held_s - 1.9) < 1e-9
    assert c.update(StandingSample(**GOOD), 2.0).standing

def test_each_threshold_names_its_check():
    c = StandingCheck(G1_SPEC.standing)
    bad = {"walker": dict(walker_live=False), "height": dict(torso_height=0.70), "tilt": dict(pitch=0.2),
           "gyro": dict(gyro_norm=0.3), "legs": dict(leg_dq_max=0.8), "stale": dict(lowstate_age_s=0.5)}
    for name, over in bad.items():
        r = c.update(StandingSample(**{**GOOD, **over}), 0.0)
        assert r.failing == (name,), name
    r = c.update(StandingSample(**{**GOOD, "roll": 0.2, "gyro_norm": 0.9}), 0.0)
    assert r.failing == ("tilt", "gyro")

def test_failure_restarts_the_hold():
    c = StandingCheck(G1_SPEC.standing)
    c.update(StandingSample(**GOOD), 0.0)
    c.update(StandingSample(**GOOD), 1.5)
    c.update(StandingSample(**{**GOOD, "gyro_norm": 0.5}), 1.6)
    r = c.update(StandingSample(**GOOD), 1.7)
    assert not r.standing and r.held_s == 0.0
    assert c.update(StandingSample(**GOOD), 3.7).standing

def test_height_band_is_symmetric_about_stand_height():
    c = StandingCheck(G1_SPEC.standing)
    assert c.update(StandingSample(**{**GOOD, "torso_height": 0.78 + 0.059}), 0.0).failing == ()
    c.reset()
    assert c.update(StandingSample(**{**GOOD, "torso_height": 0.78 - 0.061}), 0.0).failing == ("height",)

def test_nan_inputs_fail_closed():
    nan = float("nan")
    c = StandingCheck(G1_SPEC.standing)
    for key, name in (("torso_height", "height"), ("roll", "tilt"), ("pitch", "tilt"), ("gyro_norm", "gyro"),
                      ("leg_dq_max", "legs"), ("lowstate_age_s", "stale")):
        assert c.update(StandingSample(**{**GOOD, key: nan}), 0.0).failing == (name,), key
