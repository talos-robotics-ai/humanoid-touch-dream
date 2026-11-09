import math
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
