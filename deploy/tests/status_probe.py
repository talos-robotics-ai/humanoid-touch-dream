"""Record rt/htd/supervisor and rt/htd/policy_status straight from DDS (no UI server in the path).

    python deploy/tests/status_probe.py --iface lo --seconds 20 --out status.json
    python deploy/tests/status_probe.py --iface lo --count_writers rt/lowcmd

The first form writes the same JSON as ws_probe.py: one row per SupervisorStatus with the
latest PolicyStatus. The second prints how many live DDS writers the topic has, after
`--seconds` (default 3) of discovery, and writes nothing.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from ws_probe import row_from, write  # noqa: E402


def count_writers(participant, topic, seconds):
    from cyclonedds.builtin import (BuiltinDataReader, BuiltinTopicDcpsPublication, InstanceState,
                                    ReadCondition, SampleState, ViewState)
    rd = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    time.sleep(seconds)
    alive = ReadCondition(rd, SampleState.Any | ViewState.Any | InstanceState.Alive)
    return sum(1 for s in rd.read(N=10000, condition=alive) if getattr(s, "topic_name", None) == topic)


def record(participant, seconds):
    import dataclasses
    from deploy import dds
    from deploy.msgs import TOPIC_POLICY_STATUS, TOPIC_SUPERVISOR, PolicyStatus, SupervisorStatus
    r_sup = dds.reader(participant, TOPIC_SUPERVISOR, SupervisorStatus)
    r_pol = dds.reader(participant, TOPIC_POLICY_STATUS, PolicyStatus)
    rows, pol = [], None
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for s in r_pol.take(N=8):
            if isinstance(s, PolicyStatus):
                pol = dataclasses.asdict(s)
        for s in r_sup.take(N=8):
            if isinstance(s, SupervisorStatus):
                rows.append(row_from(dataclasses.asdict(s), pol, time.time()))
        time.sleep(0.01)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iface", default=None, help="network interface for CycloneDDS (lo for the sim)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--count_writers", metavar="TOPIC", default=None)
    args = ap.parse_args(argv)
    from deploy import dds
    dds.configure_interface(args.iface)
    from cyclonedds.domain import DomainParticipant
    dp = DomainParticipant(0)
    if args.count_writers:
        print(count_writers(dp, args.count_writers, 3.0 if args.seconds is None else args.seconds))
        return 0
    if args.seconds is None or args.out is None:
        ap.error("--seconds and --out are required unless --count_writers is given")
    t0 = time.time()
    rows = record(dp, args.seconds)
    write(args.out, rows, {"source": "dds", "iface": args.iface, "seconds": args.seconds,
                           "start": t0, "end": time.time()})
    print(f"status_probe: {len(rows)} supervisor samples -> {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
