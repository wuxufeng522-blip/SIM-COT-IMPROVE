from __future__ import annotations

import copy
import json

import pytest

from reliable_simcot.m1_training import atomic_json, sha256_file, verify_schedule


def test_atomic_json_and_sha256_are_stable(tmp_path) -> None:
    path = tmp_path / "result.json"
    atomic_json(path, {"value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    assert len(sha256_file(path)) == 64


def test_m1_configs_freeze_equal_budget() -> None:
    common = json.loads(
        open("configs/reliable_simcot/m1_common.json", encoding="utf-8").read()
    )
    r010 = json.loads(
        open("configs/reliable_simcot/r010_coconut.json", encoding="utf-8").read()
    )
    r011 = json.loads(
        open("configs/reliable_simcot/r011_simcot.json", encoding="utf-8").read()
    )
    assert common["updates"] == 1024
    assert common["gradient_accumulation_steps"] == 8
    assert r010["variant"] == "coconut"
    assert r011["variant"] == "simcot"
    assert set(r010) == set(r011)


def test_frozen_schedule_detects_mutation() -> None:
    common = json.loads(
        open("configs/reliable_simcot/m1_common.json", encoding="utf-8").read()
    )
    schedule = json.loads(open(common["schedule_path"], encoding="utf-8").read())
    verify_schedule(schedule)
    assert schedule["audit_manifest_sha256"] == common["audit_manifest_sha256"]
    assert all(
        row["reason"] == "natural_audit_question"
        for row in schedule["rejected_candidates"]
    )
    mutated = copy.deepcopy(schedule)
    mutated["entries"][0]["idx"] += 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_schedule(mutated)
