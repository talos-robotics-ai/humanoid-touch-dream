#!/usr/bin/env python3
"""Verify that the walker code this module runs is the pinned teleop code.

    python g1_walk/check_pin.py                     # the submodule against pins.env
    python g1_walk/check_pin.py --against ~/teleop-cbf-perception

Fatal checks: the submodule is at pins.env's TELEOP_COMMIT and this repository's index
records the same commit; the walker's deploy set (tracked files under teleop/ and
assets/ plus third_party/g1_walk, what the teleop's tools/g1_jetson_walker.sh pushes
to the robot) has no local change and no untracked file; third_party/g1_walk is at the
commit the pinned tree records. unitree_sdk2py comes from the environment: a git
checkout is compared with the pinned gitlink, anything else is reported as unverified.
--against PATH also compares the deploy set byte for byte with another checkout of
teleop-cbf-perception and lists every file that differs. Exit status 1 on any failure.
"""
import argparse
import filecmp
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SUB_REL = "g1_walk/teleop-cbf-perception"
SUB = os.path.join(REPO, SUB_REL)
DEPLOY_DIRS = ("teleop", "assets")
NESTED = "third_party/g1_walk"
SDK_NESTED = "third_party/unitree_sdk2_python"


def read_pins(path=os.path.join(HERE, "pins.env")):
    pins = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                pins[key.strip()] = value.strip()
    return pins


def git(cwd, *args):
    return subprocess.run(["git", "-C", cwd, *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def gitlink(repo, tree, path):
    """The commit a tree (or, with tree None, the index) records for a
    submodule path; None when it records none."""
    out = (git(repo, "ls-files", "-s", "--", path) if tree is None
           else git(repo, "ls-tree", tree, "--", path))
    fields = out.split()
    if not fields or fields[0] != "160000":
        return None
    return fields[1] if tree is None else fields[2]


def dirty(repo, *paths):
    """Modified, staged or untracked (not ignored) paths under ``paths``."""
    out = git(repo, "status", "--porcelain", "--", *paths)
    return [line for line in out.splitlines() if line]


def deploy_files():
    """(path relative to the teleop checkout) for the walker's deploy set."""
    files = git(SUB, "ls-files", "--", *DEPLOY_DIRS).splitlines()
    nested = os.path.join(SUB, NESTED)
    files += [os.path.join(NESTED, f) for f in git(nested, "ls-files").splitlines()]
    return files


def check_submodule(pins):
    errors = []
    if not os.path.exists(os.path.join(SUB, ".git")):
        return ["%s is not checked out: run bash g1_walk/setup.sh" % SUB_REL]
    want = pins["TELEOP_COMMIT"]
    head = git(SUB, "rev-parse", "HEAD")
    if head != want:
        errors.append("%s is at %s, pins.env says %s" % (SUB_REL, head, want))
    recorded = gitlink(REPO, None, SUB_REL)
    if recorded != want:
        errors.append("this repository's index records %s at %s, pins.env says %s"
                      % (SUB_REL, recorded, want))
    changed = dirty(SUB, *DEPLOY_DIRS)
    if changed:
        errors.append("the pinned walker code has local changes:\n  "
                      + "\n  ".join(changed))
    nested = os.path.join(SUB, NESTED)
    expect = gitlink(SUB, "HEAD", NESTED)
    if not os.path.exists(os.path.join(nested, ".git")):
        errors.append("%s is not initialised: run bash g1_walk/setup.sh" % NESTED)
    else:
        got = git(nested, "rev-parse", "HEAD")
        if got != expect:
            errors.append("%s is at %s, the pinned tree records %s" % (NESTED, got, expect))
        if dirty(nested, "."):
            errors.append("%s has local changes" % NESTED)
    return errors


def sdk_report():
    """One line on where unitree_sdk2py comes from, and an error or None."""
    spec = importlib.util.find_spec("unitree_sdk2py")
    if spec is None or spec.origin is None:
        return "unitree_sdk2py: not installed", "unitree_sdk2py is not importable"
    root = os.path.dirname(os.path.dirname(spec.origin))
    expect = gitlink(SUB, "HEAD", SDK_NESTED)
    try:
        got = git(root, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unitree_sdk2py: %s (not a git checkout: unverified)" % root, None
    if got != expect:
        return ("unitree_sdk2py: %s at %s" % (root, got),
                "unitree_sdk2py is at %s, the pinned tree records %s" % (got, expect))
    return "unitree_sdk2py: %s at the pinned %s" % (root, got[:8]), None


def compare(other):
    """Deploy-set files whose bytes differ from (or are missing in) ``other``."""
    differ = []
    for rel in deploy_files():
        mine, theirs = os.path.join(SUB, rel), os.path.join(other, rel)
        if os.path.islink(mine) or os.path.isdir(mine):
            continue
        if not os.path.isfile(theirs) or not filecmp.cmp(mine, theirs, shallow=False):
            differ.append(rel)
    return differ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--against", metavar="PATH",
                    help="another teleop-cbf-perception checkout to compare the deploy set with")
    args = ap.parse_args(argv)
    pins = read_pins()
    errors = check_submodule(pins)
    if not errors:
        print("ok: %s at %s, deploy set clean, %s at %s"
              % (SUB_REL, pins["TELEOP_COMMIT"][:8], NESTED,
                 git(os.path.join(SUB, NESTED), "rev-parse", "HEAD")[:8]))
        line, err = sdk_report()
        print(line)
        if err:
            errors.append(err)
    if args.against and not errors:
        differ = compare(os.path.abspath(os.path.expanduser(args.against)))
        if differ:
            errors.append("%d file(s) of the deploy set differ in %s:\n  %s"
                          % (len(differ), args.against, "\n  ".join(differ)))
        else:
            print("ok: the deploy set is byte-identical in %s" % args.against)
    for e in errors:
        print("FAILED: " + e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
