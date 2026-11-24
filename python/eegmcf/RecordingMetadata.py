"""Shared recording identity and provenance helpers for DAT, FIF and PDF workflows"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


UNKNOWN = "N/A"
PACKAGE_VERSION = "1.1.1"
HELPER_DIR = Path(__file__).resolve().parent / "helper"


def known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "n/a", "unknown", "none", "nan"}


def clean(value: Any) -> str:
    return str(value).strip() if known(value) else UNKNOWN


def parameter_value(params: Any, key: str) -> str:
    """Extract the scalar value of a BCI2000 parameter rather than its definition"""
    try:
        param = params[key]
    except (KeyError, TypeError, AttributeError, IndexError):
        return UNKNOWN
    for attribute in ("val", "value", "Value"):
        value = getattr(param, attribute, None)
        if value is not None and not callable(value):
            param = value
            break
    if isinstance(param, (list, tuple)) and len(param) == 1:
        param = param[0]
    return clean(param)


def parse_recording_time(value: Any) -> tuple[str, str, datetime | None]:
    """Preserve recording-local date and time without inventing a timezone"""
    if not known(value):
        return UNKNOWN, UNKNOWN, None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        timestamp = dt.isoformat(timespec="seconds") if len(text) > 10 else UNKNOWN
        return dt.date().isoformat(), timestamp, dt if timestamp != UNKNOWN and dt.utcoffset() is not None else None
    except ValueError:
        pass
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Z %Y", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.date().isoformat(), dt.isoformat(timespec="seconds") if "%H" in fmt else UNKNOWN, None
        except ValueError:
            continue
    return UNKNOWN, UNKNOWN, None


def recording_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.lower().endswith(".fif.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    return re.sub(r"_cleaned_raw$", "", name, flags=re.IGNORECASE)


def identity_from_name(name: str) -> dict[str, str]:
    def field(key: str) -> str:
        match = re.search(rf"(?:^|_){key}-([^_]+)", name, re.IGNORECASE)
        return clean(match.group(1) if match else None)
    return {"recording_id": clean(name), "subject_id": field("sub"), "session_id": field("ses"), "run_id": field("run"), "task": field("task")}


def metadata_from_dat(path: str | Path, params: Any) -> dict[str, Any]:
    storage = parameter_value(params, "StorageTime")
    day, stamp, _ = parse_recording_time(storage)
    data = identity_from_name(recording_stem(path))
    for key, parameter in (("subject_id", "SubjectName"), ("session_id", "SubjectSession"), ("run_id", "SubjectRun")):
        if not known(data[key]):
            data[key] = parameter_value(params, parameter)
    data.update(recording_date=day, date_test=day, recording_timestamp=stamp,
                recording_date_source="BCI2000 StorageTime" if known(day) else UNKNOWN,
                storage_time=storage, source_file=str(Path(path).resolve()))
    return data


def metadata_from_raw(raw: Any, path: str | Path, saved: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read sidecar data first, then fall back to acquisition metadata in FIF"""
    data = dict(saved or {})
    source = data.get("source_file") if known(data.get("source_file")) else path
    defaults = identity_from_name(recording_stem(source))
    for key, value in defaults.items():
        if not known(data.get(key)):
            data[key] = value
    if not known(data.get("subject_id")):
        subject_info = raw.info.get("subject_info") or {}
        data["subject_id"] = clean(subject_info.get("his_id"))
    for candidate in (data.get("recording_timestamp"), data.get("recording_date"), data.get("date_test")):
        day, stamp, _ = parse_recording_time(candidate)
        if known(day):
            data["recording_date"] = day
            if known(stamp):
                data["recording_timestamp"] = stamp
            if not known(data.get("recording_date_source")):
                data["recording_date_source"] = "recording sidecar"
            break
    else:
        day, stamp, _ = parse_recording_time(raw.info.get("meas_date"))
        data.update(recording_date=day, recording_timestamp=stamp,
                    recording_date_source="MNE meas_date (UTC)" if known(day) else UNKNOWN)
    data["date_test"] = data["recording_date"]
    return data


def apply_overrides(data: dict[str, Any], args: Any) -> dict[str, Any]:
    """Explicit operator values override automatically extracted metadata"""
    data = dict(data)
    for key in ("recording_id", "subject_id", "session_id", "run_id", "age_at_test", "condition"):
        value = getattr(args, key, None)
        if value is not None:
            data[key] = clean(value)
    value = getattr(args, "recording_date", None)
    if value is not None:
        day, stamp, _ = parse_recording_time(value)
        if not known(day):
            raise ValueError("--recording-date must be YYYY-MM-DD or an ISO 8601 timestamp")
        data.update(recording_date=day, date_test=day, recording_timestamp=stamp, recording_date_source="operator override")
    return data


def add_metadata_arguments(parser: Any) -> None:
    parser.add_argument("--recording-date", help="Override recording date (YYYY-MM-DD or ISO 8601 timestamp)")
    parser.add_argument("--recording-id", help="Override recording identifier (default: original recording stem)")
    parser.add_argument("--subject-id", help="Override pseudonymous subject ID")
    parser.add_argument("--session-id", help="Override session ID")
    parser.add_argument("--run-id", help="Override run ID")
    parser.add_argument("--age-at-test", help="Age at recording; retained in the cleaned Raw sidecar")
    parser.add_argument("--condition", help="Explicit condition label; unknown unless provided or saved")


def spatial_filter_label(history: Mapping[str, Any] | None) -> str:
    if history is None:
        return UNKNOWN
    if "spatial_filter" in history or any(isinstance(step, dict) and step.get("step") == "spatialfilter" and step.get("status") == "done" for step in history.get("step_log", [])):
        return "Surface Laplacian"
    return "Not applied"
