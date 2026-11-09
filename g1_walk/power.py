"""Whether this machine is set up to run control loops in real time.

On a laptop, battery power or a power-saving platform profile cuts the CPU
clock (to about 1.2 GHz on the RTX 5070 laptop this module was tested on). The
walker's 50 Hz policy, the arm IK and the hand driver then miss their periods
and the state the policy reads arrives late, so the robot must not be driven
that way. Reads Linux sysfs only; on a machine without these entries (a
desktop, a container) it reports nothing.
"""
import glob
import os

SLOW_PROFILES = ("low-power", "quiet", "cool")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def power_state():
    """``(on_mains, platform_profile)``; each None when the machine does not say."""
    on_mains = None
    for supply in glob.glob("/sys/class/power_supply/*"):
        if _read(os.path.join(supply, "type")) == "Mains":
            online = _read(os.path.join(supply, "online"))
            if online is not None:
                on_mains = bool(on_mains) or online == "1"
    return on_mains, _read("/sys/firmware/acpi/platform_profile")


def power_problems():
    """Human-readable reasons this machine is not ready for real-time control."""
    on_mains, profile = power_state()
    problems = []
    if on_mains is False:
        problems.append("running on battery: plug in the charger")
    if profile in SLOW_PROFILES:
        problems.append("platform profile is '%s': switch to performance "
                        "(powerprofilesctl set performance)" % profile)
    return problems
