#!/usr/bin/env python3
"""Generate a per-run deployment config from each run's saved ``env.yaml``.

For every trained run, this copies the template deploy config
(``deploy/configs/g1_student_htd.yaml``) and overrides ONLY the 15 leg+waist
``kps``/``kds`` (and the matching joint-map comment) with the actuator PD gains
that run was trained/distilled under, read from ``<run>/params/env.yaml``.
Everything else -- command ranges, scales, ``waist_pitch_limits``, arm gains,
``policy_path`` -- is copied through byte-for-byte.

Run it where ``logs/`` lives (e.g. the cluster), so it reads the CURRENT gains
rather than a stale local copy. Needs only PyYAML (no Isaac deps)::

    python tools/gen_deploy_config.py --all-versions v4 v5 v6
    python tools/gen_deploy_config.py --run ci20-60_wrp150-4_hpr150-3_fcs0p5_e12288_s1_v6

Writes ``deploy/configs/<run>.yaml`` for each run.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import yaml

TEMPLATE = "deploy/configs/g1_student_htd.yaml"
LOGROOT = "logs/g1_flat"

# Deploy joint order (indices 0-14 in kps/kds) -> sim joint name used for gain lookup.
# Gains are bilaterally symmetric, so left/right resolve to the same regex value.
JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


class _TagIgnoringLoader(yaml.SafeLoader):
    """SafeLoader that drops Isaac Lab's Python object tags (e.g. slice) instead of
    failing on them -- we only read the plain actuator stiffness/damping dicts."""


def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TagIgnoringLoader.add_multi_constructor("", _ignore_unknown)


def _load_actuators(env_yaml: str) -> dict:
    with open(env_yaml) as f:
        data = yaml.load(f, Loader=_TagIgnoringLoader)
    return data["scene"]["robot"]["actuators"]


def _field_for_joint(field, joint: str):
    """Resolve a stiffness/damping entry (scalar or {regex: value} dict) for one joint."""
    if field is None:
        return None
    if isinstance(field, dict):
        val = None
        for key, value in field.items():
            if re.fullmatch(str(key), joint):
                val = value  # last matching regex wins, mirroring config layering
        return val
    return field  # scalar applies to every joint in the group


def _gain_for_joint(actuators: dict, joint: str):
    for group in actuators.values():
        if not isinstance(group, dict):
            continue
        exprs = group.get("joint_names_expr") or []
        if isinstance(exprs, str):
            exprs = [exprs]
        if not any(re.fullmatch(str(e), joint) for e in exprs):
            continue
        kp = _field_for_joint(group.get("stiffness"), joint)
        kd = _field_for_joint(group.get("damping"), joint)
        if kp is not None and kd is not None:
            return kp, kd
    raise KeyError(f"no actuator gains found for joint '{joint}'")


def _num(x):
    f = float(x)
    return int(f) if f.is_integer() else round(f, 6)


def _fmt_list(vals) -> str:
    return "[" + ", ".join(str(v) for v in vals) + "]"


def gen_one(run: str, template_lines: list[str]):
    env_yaml = os.path.join(LOGROOT, run, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        raise FileNotFoundError(env_yaml)
    actuators = _load_actuators(env_yaml)

    kps, kds = [], []
    for joint in JOINTS:
        kp, kd = _gain_for_joint(actuators, joint)
        kps.append(_num(kp))
        kds.append(_num(kd))

    out = []
    for line in template_lines:
        if re.match(r"^kps:\s*\[", line):
            out.append(f"kps:               {_fmt_list(kps)}\n")
        elif re.match(r"^kds:\s*\[", line):
            out.append(f"kds:               {_fmt_list(kds)}\n")
        else:
            m = re.match(r"^# (\d+):\s*(\S+)\s+kp:\s*\S+\s+kd:\s*\S+(.*)$", line)
            if m and int(m.group(1)) < len(JOINTS):
                i = int(m.group(1))
                out.append(f"# {i}: {m.group(2)}  kp: {kps[i]}  kd: {kds[i]}{m.group(3)}\n")
            else:
                out.append(line)
    return kps, kds, "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[], help="Run folder name(s) under logs/g1_flat/.")
    ap.add_argument("--all-versions", nargs="*", default=[], metavar="vN",
                    help="Expand to every logs/g1_flat/*_vN folder.")
    ap.add_argument("--template", default=TEMPLATE, help=f"Template deploy config (default: {TEMPLATE}).")
    ap.add_argument("--outdir", default="deploy/configs", help="Output directory (default: deploy/configs).")
    args = ap.parse_args()

    runs = list(args.run)
    for v in args.all_versions:
        runs += [os.path.basename(p) for p in sorted(glob.glob(f"{LOGROOT}/*_{v}")) if os.path.isdir(p)]
    runs = sorted(dict.fromkeys(runs))
    if not runs:
        sys.exit("No runs given. Use --run <name> ... or --all-versions v4 v5 v6")

    with open(args.template) as f:
        template_lines = f.readlines()

    os.makedirs(args.outdir, exist_ok=True)
    ok = 0
    for run in runs:
        try:
            kps, kds, text = gen_one(run, template_lines)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[skip] {run}: {exc}")
            continue
        outpath = os.path.join(args.outdir, f"{run}.yaml")
        with open(outpath, "w") as f:
            f.write(text)
        ok += 1
        print(f"[ok] {outpath}")
        print(f"      kps={kps}")
        print(f"      kds={kds}")
    print(f"\nWrote {ok}/{len(runs)} deploy configs to {args.outdir}/")


if __name__ == "__main__":
    main()
