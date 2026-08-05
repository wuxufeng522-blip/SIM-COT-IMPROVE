from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import pytest

from reliable_simcot.audit import question_id
from reliable_simcot.labels import ReliabilityRow
from reliable_simcot.official_adapter import OfficialExample
from reliable_simcot.reliability_data import (
    build_rows_for_examples,
    load_sealed_rows,
)


def test_invalid_rows_cannot_define_utility() -> None:
    with pytest.raises(ValueError, match="undefined"):
        ReliabilityRow(
            variant_id="v",
            question_id="q",
            source_index=0,
            split="head_train",
            question="question",
            answer="100",
            step_index=0,
            trajectory_steps=1,
            prefix_steps=(),
            clean_step="<<20*5=100>>",
            candidate_step="<<20*5=101>>",
            family="numeric_error",
            template_id="t",
            pair_id="p",
            y_valid=0,
            y_utility=0,
            metadata={},
        )


def test_variants_share_question_split_and_pair_position() -> None:
    example = OfficialExample(
        idx=3,
        question="There are 20 groups of 5. How many?",
        steps=("<<20*5=100>>", "<<100-30=70>>"),
        answer="70",
    )
    qid = question_id(example.question)
    development, sealed, diagnostics = build_rows_for_examples(
        [example],
        split_by_question={qid: "head_audit"},
    )
    assert development
    assert sealed
    assert diagnostics["eligible_clean_steps"] == 2
    assert {row.question_id for row in development} == {qid}
    assert {row.split for row in development} == {"head_audit"}
    for row in development:
        assert row.pair_id == f"{qid}:{row.step_index}"
        assert len(row.prefix_steps) == row.step_index


def test_sealed_rows_require_frozen_head_binding(tmp_path: Path) -> None:
    sealed_path = tmp_path / "sealed.jsonl"
    sealed_payload = b'{"variant_id":"v"}\n'
    sealed_path.write_bytes(sealed_payload)
    sealed_sha = sha256(sealed_payload).hexdigest()
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "sealed_stress": {
                    "path": str(sealed_path),
                    "sha256": sealed_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    head_manifest = tmp_path / "head_manifest.json"
    head_manifest.write_text(
        json.dumps({"reliability_head_frozen": False}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="before head freeze"):
        load_sealed_rows(
            dataset_manifest_path=dataset_manifest,
            frozen_head_manifest_path=head_manifest,
        )

    head_manifest.write_text(
        json.dumps(
            {
                "reliability_head_frozen": True,
                "sealed_stress_sha256": sealed_sha,
            }
        ),
        encoding="utf-8",
    )
    assert load_sealed_rows(
        dataset_manifest_path=dataset_manifest,
        frozen_head_manifest_path=head_manifest,
    ) == [{"variant_id": "v"}]
