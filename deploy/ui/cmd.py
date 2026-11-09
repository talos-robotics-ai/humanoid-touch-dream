"""Publish one UiCommand from the command line: python deploy/ui/cmd.py [--iface IFACE] start|stop|rearm|reset"""
import argparse
import itertools
import os
import random
import sys
import threading
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from deploy import dds  # noqa: E402
from deploy.msgs import TOPIC_UI_COMMAND, UI_COMMANDS, UiCommand  # noqa: E402

MATCH_TIMEOUT_S = 2.0


class NonceSource:
    """UiCommand nonces, unique across restarts: wall-clock ms (30 bits) plus a counter, within int32.
    The counter starts at a random offset, so two processes started in the same millisecond differ."""

    def __init__(self):
        self._count = itertools.count(random.getrandbits(30))
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            return ((int(time.time() * 1000) & 0x3fffffff) + next(self._count)) & 0x7fffffff


def matched_writer(participant, timeout_s: float = MATCH_TIMEOUT_S):
    """A UiCommand writer, and whether a reader matched it within timeout_s: a best-effort
    write before the controller's reader matches is lost."""
    from cyclonedds.core import Listener
    from cyclonedds.pub import DataWriter
    from cyclonedds.topic import Topic
    matched = threading.Event()

    def on_matched(writer, status):
        if status.current_count > 0:
            matched.set()

    w = DataWriter(participant, Topic(participant, TOPIC_UI_COMMAND, UiCommand), qos=dds.telemetry_qos(),
                   listener=Listener(on_publication_matched=on_matched))
    return w, matched.wait(timeout_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", default=None, help="network interface for CycloneDDS")
    ap.add_argument("command", help="|".join(UI_COMMANDS))
    args = ap.parse_args(argv)
    if args.command not in UI_COMMANDS:
        print(f"unknown command {args.command!r}; expected one of {', '.join(UI_COMMANDS)}", file=sys.stderr)
        return 2
    dds.configure_interface(args.iface)
    from cyclonedds.domain import DomainParticipant
    w, matched = matched_writer(DomainParticipant(0))
    if not matched:
        print(f"no reader of {TOPIC_UI_COMMAND} matched within {MATCH_TIMEOUT_S:.0f} s: is the controller "
              "running with --supervisor on, on the same --iface?", file=sys.stderr)
        return 1
    time.sleep(0.05)    # the reader's side of the match
    nonce = NonceSource().next()
    w.write(UiCommand(timestamp=time.time(), nonce=nonce, command=args.command, source="cli"))
    time.sleep(0.1)     # let the sample leave before the participant is torn down
    print(nonce)
    return 0


if __name__ == "__main__":
    sys.exit(main())
