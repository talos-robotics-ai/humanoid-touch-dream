from deploy import msgs

def test_topics_are_distinct_and_namespaced():
    topics = [msgs.TOPIC_SUPERVISOR, msgs.TOPIC_POLICY_STATUS, msgs.TOPIC_PREDICTED_CHUNK,
              msgs.TOPIC_CAMERA, msgs.TOPIC_UI_COMMAND]
    assert len(set(topics)) == 5 and all(t.startswith("rt/htd/") for t in topics)

def test_messages_round_trip_through_cdr():
    chunk = msgs.PredictedChunk(timestamp=1.5, t_obs=30, horizon=2, interval=45, inference_ms=6.6,
                                body_pose=[0.0] * 12, left_eef_to_body=[0.1] * 12, right_eef_to_body=[0.2] * 12,
                                left_eef_link=[0.3] * 30, right_eef_link=[0.4] * 30, velocity_commands=[0.5] * 6)
    assert msgs.PredictedChunk.deserialize(chunk.serialize()) == chunk
    cmd = msgs.UiCommand(timestamp=2.0, nonce=7, command="start", source="ui")
    assert msgs.UiCommand.deserialize(cmd.serialize()).command == "start"
    frame = msgs.CameraFrame(timestamp=3.0, width=4, height=2, jpeg=list(b"\xff\xd8\x00\xff"))
    assert bytes(msgs.CameraFrame.deserialize(frame.serialize()).jpeg) == b"\xff\xd8\x00\xff"

def test_ui_commands_constant():
    assert msgs.UI_COMMANDS == ("start", "stop", "rearm", "reset")

def test_status_messages_round_trip_through_cdr():
    sup = msgs.SupervisorStatus(
        timestamp=1.0, state="READY", since_s=2.5, reason="standing 2.0 s", walker_live=True, standing=True,
        standing_for_s=2.0, policy_ready=True, frames_fresh=True, walk_enabled=True, manual_walk_enabled=False,
        fault="", gating=["press Start"], stick_lx=0.0, stick_ly=0.5, stick_rx=-0.25, stick_ry=0.0,
        buttons=["L1", "start"], manual_vx=0.1, manual_wz=-0.2, tick_p50_ms=0.25, tick_max_ms=1.5,
        log_line="HTDPOLICY READY TO DEPLOY",
        recent_log=["[supervisor] BOOT -> STABILIZING: walker live", "HTDPOLICY READY TO DEPLOY"])
    assert msgs.SupervisorStatus.deserialize(sup.serialize()) == sup
    pol = msgs.PolicyStatus(timestamp=2.0, ready=False, publishing=True, inference_ms=9.75, chunks=12,
                            overrun_steps=3, robot_state_age_s=0.02, eef_state_age_s=0.03, camera_age_s=1e9,
                            loop_p50_ms=1.25, loop_max_ms=4.0, interval=45, horizon=60,
                            checkpoint="/runs/x/policy_last.ckpt", detail="no camera frame")
    assert msgs.PolicyStatus.deserialize(pol.serialize()) == pol
    assert msgs.RECENT_LOG_LINES == 12
