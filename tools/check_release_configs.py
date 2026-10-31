#!/usr/bin/env python3
"""Lightweight release checks that do not import Isaac Lab or robot SDKs."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal release-check environments
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_CONFIG_DIR = ROOT / "deploy" / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a YAML mapping")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple top-level YAML shape used by deploy/configs/*.yaml."""

    entries: list[tuple[str, str]] = []
    current_key: str | None = None
    current_value: list[str] = []
    bracket_balance = 0

    def flush() -> None:
        nonlocal current_key, current_value, bracket_balance
        if current_key is not None:
            entries.append((current_key, " ".join(current_value).strip()))
        current_key = None
        current_value = []
        bracket_balance = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and ":" in line and bracket_balance == 0:
            flush()
            key, value = line.split(":", 1)
            current_key = key.strip()
            current_value = [value.strip()]
        elif current_key is not None:
            current_value.append(line.strip())
        else:
            raise ValueError(f"unsupported YAML line: {raw_line}")

        joined = " ".join(current_value)
        bracket_balance = joined.count("[") - joined.count("]")

    flush()
    return {key: _parse_simple_yaml_value(value) for key, value in entries}


def _parse_simple_yaml_value(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if value.startswith("{") and value.endswith("}"):
        return _parse_simple_yaml_mapping(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml_mapping(value: str) -> dict[str, Any]:
    body = value[1:-1].strip()
    if not body:
        return {}

    result: dict[str, Any] = {}
    start = 0
    bracket_depth = 0
    parts: list[str] = []
    for idx, char in enumerate(body):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            parts.append(body[start:idx].strip())
            start = idx + 1
    parts.append(body[start:].strip())

    for part in parts:
        key, raw_value = part.split(":", 1)
        result[key.strip()] = _parse_simple_yaml_value(raw_value.strip())
    return result


def _check_len(errors: list[str], cfg_name: str, key: str, value: Any, expected: int) -> None:
    if not isinstance(value, list):
        errors.append(f"{cfg_name}: `{key}` must be a list")
        return
    if len(value) != expected:
        errors.append(f"{cfg_name}: `{key}` has {len(value)} entries, expected {expected}")


def check_deploy_config(path: Path, errors: list[str]) -> None:
    cfg = _load_yaml(path)
    cfg_name = path.relative_to(ROOT).as_posix()
    start_error_count = len(errors)

    required = [
        "control_dt",
        "msg_type",
        "imu_type",
        "lowcmd_topic",
        "lowstate_topic",
        "policy_path",
        "history_length",
        "joint2motor_idx",
        "kps",
        "kds",
        "default_joint_pos",
        "action_scale",
        "command_scale",
        "num_actions",
        "num_obs",
        "command_range",
    ]
    for key in required:
        if key not in cfg:
            errors.append(f"{cfg_name}: missing required key `{key}`")

    if len(errors) > start_error_count:
        return

    num_actions = cfg["num_actions"]
    if not isinstance(num_actions, int) or num_actions <= 0:
        errors.append(f"{cfg_name}: `num_actions` must be a positive integer")
        return

    history_length = cfg["history_length"]
    if not isinstance(history_length, int) or history_length <= 0:
        errors.append(f"{cfg_name}: `history_length` must be a positive integer")

    command_range = cfg["command_range"]
    if not isinstance(command_range, dict) or not command_range:
        errors.append(f"{cfg_name}: `command_range` must be a non-empty mapping")
        return

    command_dim = len(command_range)
    for key in ("joint2motor_idx", "kps", "kds", "default_joint_pos"):
        _check_len(errors, cfg_name, key, cfg[key], num_actions)

    _check_len(errors, cfg_name, "command_scale", cfg["command_scale"], command_dim)

    expected_num_obs = 3 + 3 + command_dim + 3 * num_actions
    if cfg["num_obs"] != expected_num_obs:
        errors.append(f"{cfg_name}: `num_obs` is {cfg['num_obs']}, expected {expected_num_obs}")

    arm_keys = ["arm_joint2motor_idx", "arm_kps", "arm_kds", "arm_default_joint_pos"]
    present_arm_keys = [key for key in arm_keys if key in cfg]
    if present_arm_keys and len(present_arm_keys) != len(arm_keys):
        missing = sorted(set(arm_keys) - set(present_arm_keys))
        errors.append(f"{cfg_name}: incomplete arm config, missing {missing}")
    elif present_arm_keys:
        arm_count = len(cfg["arm_joint2motor_idx"])
        for key in arm_keys[1:]:
            _check_len(errors, cfg_name, key, cfg[key], arm_count)


def check_pyproject(errors: list[str]) -> None:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        data = tomllib.load(file)

    project = data.get("project", {})
    if project.get("name") != "isaaclab-decoupled-wbc":
        errors.append("pyproject.toml: project name must be `isaaclab-decoupled-wbc`")
    if not project.get("version"):
        errors.append("pyproject.toml: project version is required")
    if "legged_lab*" not in data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("include", []):
        errors.append("pyproject.toml: setuptools package include must contain `legged_lab*`")


def check_python_syntax(errors: list[str]) -> None:
    for rel_path in ("legged_lab/utils/amass.py", "legged_lab/mdp/commands.py"):
        path = ROOT / rel_path
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
        except SyntaxError as exc:
            errors.append(f"{rel_path}: syntax error: {exc}")


def main() -> int:
    errors: list[str] = []

    check_pyproject(errors)
    check_python_syntax(errors)

    for path in sorted(DEPLOY_CONFIG_DIR.glob("*.yaml")):
        try:
            check_deploy_config(path, errors)
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"{path.relative_to(ROOT)}: failed to parse: {exc}")

    if errors:
        print("Release config checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Release config checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
