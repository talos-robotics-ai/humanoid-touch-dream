#!/usr/bin/env bash
# Run the pinned teleop's G1 walker (robot_control/g1_jetson_walker.py) on this machine,
# its network on this machine's GPU. It publishes rt/lowcmd for all 29 motors from the
# moment it prints READY, holding the measured pose until the controller's handover.
#
#   bash g1_walk/run_walker.sh --iface lo              # against the MuJoCo sim host
#   bash g1_walk/run_walker.sh --iface <robot NIC>     # the robot, from this laptop
#
# Options: --iface IFACE (default PSI_DDS_IFACE), --port PORT (PSI_G1_JETSON_PORT or
# 5591), --device cuda|cpu (PSI_G1_WALK_DEVICE or cuda), --python PY (python on PATH).
# Stopping it (Ctrl-C) leaves the motors with no command: have the robot on its gantry
# or sitting first (README.md, "Stopping").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${PSI_DDS_IFACE:-}"
PORT="${PSI_G1_JETSON_PORT:-5591}"
DEVICE="${PSI_G1_WALK_DEVICE:-cuda}"
PY="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface) IFACE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$IFACE" ]] || { echo "FAILED: no DDS interface: pass --iface (lo for the sim)" >&2; exit 2; }

"$PY" "$HERE/check_pin.py" >&2 || { echo "FAILED: the walker code is not the pinned code (above)" >&2; exit 1; }

cd "$HERE/teleop-cbf-perception/teleop"
export PSI_DDS_IFACE="$IFACE" PSI_G1_JETSON_PORT="$PORT" PSI_G1_WALK_DEVICE="$DEVICE"
exec "$PY" robot_control/g1_jetson_walker.py --iface "$IFACE" --port "$PORT"
