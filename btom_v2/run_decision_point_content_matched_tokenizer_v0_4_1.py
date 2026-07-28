"""Tokenizer-only measurement of frozen v0.4.0 content-matched prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

from .decision_point_content_matched_control_prompting_v0_4_0 import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_SELF_BELIEF,
    FIRST_ORDER_TITLE,
    MATCHED_DECISION_RECORD,
    MESSAGE_RECORD,
    PARTNER_BELIEF,
    SECOND_ORDER_TITLE,
    render_all_prompts,
    run_audit,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_content_matched_tokenizer_manifest_v0_4_1.json"
)
CALIBRATION_VERSION = "btom-v2-decision-point-content-matched-tokenizer-0.4.1"
IMMUTABLE_PARENT_SHA256 = {
    "btom_v2/decision_point_content_matched_control_manifest_v0_4_0.json": "5a45397e0d420e188b86b2520bc3a701e4dd3e84c6aa4d6d2508b821937999b6",
    "btom_v2/decision_point_content_matched_control_prompting_v0_4_0.py": "ab9c6b01308df15560db4be1a735bb5cef31a101942ff79431b38ce6b0733ccf",
    "btom_v2/decision_point_content_matched_control_design_v0_4_0.md": "581d1e769c792365e8c89dff48093de9af788fc145365d7e6b984e8f66d62c27",
    "tests/test_decision_point_content_matched_control_v0_4_0.py": "facfcfd231beaee3f460cddba5590e50251840bc8c3afb013c80ab9ef9561786",
    "btom_v2/decision_point_natural_control_prompting_v0_3_0.py": "08df09f8e5d950f0fde422993133ca59e40362bcd06ba799446951a5701c1961",
    "btom_v2/decision_point_scenarios.py": "e7a936cca701356dc2f9a5c8ab9621b5e593af932b854afe4c9a146acf1994f5",
}
STRUCTURAL_AUDIT_REQUIREMENTS = {
    "prompt_count": 16,
    "DP5_prompt_count": 6,
    "DP7_prompt_count": 10,
    "common_sections_byte_identical": True,
    "valid_actions_frozen": True,
    "observations_frozen": True,
    "raw_messages_frozen": True,
    "DP7_raw_evidence_exactly_once": True,
    "hidden_state_leak_count": 0,
    "scoring_label_leak_count": 0,
    "condition_name_leak_count": 0,
    "source_message_field_count": 0,
    "legacy_marker_count": 0,
    "primary_pair_audits_passed": True,
    "tokenizer_parity_claimed": False,
    "model_or_api_execution": False,
    "real_execution_authorized": False,
}
CONTRASTS = {
    "H1": {
        "informative": EXPLICIT_SELF_BELIEF,
        "reference": MATCHED_DECISION_RECORD,
        "cases": ("DP5a_self_belief_false", "DP5b_self_belief_current"),
        "informative_block": FIRST_ORDER_TITLE,
        "reference_block": FIRST_ORDER_TITLE,
    },
    "H2": {
        "informative": PARTNER_BELIEF,
        "reference": MESSAGE_RECORD,
        "cases": ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"),
        "informative_block": SECOND_ORDER_TITLE,
        "reference_block": SECOND_ORDER_TITLE,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_immutable_inputs() -> dict[str, Any]:
    records = {}
    failures = []
    for relative, expected in IMMUTABLE_PARENT_SHA256.items():
        actual = _sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        matches = actual == expected
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            failures.append(relative)
    if failures:
        raise RuntimeError("immutable input verification failed: " + ", ".join(failures))
    return records


def load_tokenizers(manifest: dict[str, Any]):
    # Runtime-only imports keep static checks independent of tokenizer packages.
    import tiktoken
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        load_harmony_encoding,
    )

    versions = {
        "tiktoken": importlib.metadata.version("tiktoken"),
        "openai-harmony": importlib.metadata.version("openai-harmony"),
        "pytest": importlib.metadata.version("pytest"),
    }
    if versions != manifest["dependencies"]:
        raise RuntimeError(
            f"dependency version mismatch: expected {manifest['dependencies']}, got {versions}"
        )
    raw_encoding = tiktoken.get_encoding("o200k_harmony")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def harmony_tokens(text: str) -> list[int]:
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.USER, text)
        ])
        return list(
            harmony_encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
        )

    return raw_encoding, harmony_tokens, versions


def _token_hash(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _text_measurement(
    text: str,
    raw_encode: Callable[[str], list[int]],
    harmony_encode: Callable[[str], list[int]],
) -> dict[str, Any]:
    raw_ids = list(raw_encode(text))
    harmony_ids = list(harmony_encode(text))
    return {
        "character_count": len(text),
        "utf8_byte_count": len(text.encode()),
        "line_count": len(text.splitlines()),
        "approximate_whitespace_token_count": len(text.split()),
        "raw_token_count": len(raw_ids),
        "raw_token_id_sha256": _token_hash(raw_ids),
        "harmony_token_count": len(harmony_ids),
        "harmony_token_id_sha256": _token_hash(harmony_ids),
    }


def measure_prompts(prompts, raw_encode, harmony_encode):
    prompt_records = []
    rendered_records = []
    block_records = []
    for prompt in prompts:
        sections = dict(prompt.sections)
        full = _text_measurement(prompt.prompt, raw_encode, harmony_encode)
        added_names = tuple(
            name for name in (FIRST_ORDER_TITLE, SECOND_ORDER_TITLE) if name in sections
        )
        added = {
            name: _text_measurement(sections[name], raw_encode, harmony_encode)
            for name in added_names
        }
        prompt_records.append({
            "prompt_id": prompt.prompt_id,
            "case_id": prompt.case_id,
            "family": prompt.family,
            "condition": prompt.condition,
            "prompt_sha256": hashlib.sha256(prompt.prompt.encode()).hexdigest(),
            **full,
            "added_block_token_counts": {
                name: {
                    "raw": measurement["raw_token_count"],
                    "harmony": measurement["harmony_token_count"],
                }
                for name, measurement in added.items()
            },
            "included_block_names": added_names,
            "structural_audit_status": "passed_v0_4_0_frozen_audit",
            "model_or_api_execution": False,
        })
        rendered_records.append({
            "prompt_id": prompt.prompt_id,
            "case_id": prompt.case_id,
            "family": prompt.family,
            "condition": prompt.condition,
            "prompt": prompt.prompt,
            "prompt_sha256": hashlib.sha256(prompt.prompt.encode()).hexdigest(),
        })
        for name in added_names:
            block_records.append({
                "case_id": prompt.case_id,
                "condition": prompt.condition,
                "prompt_id": prompt.prompt_id,
                "block_title": name,
                "block_text": sections[name],
                "representation_role": dict(
                    line.split("=", 1) for line in sections[name].splitlines()[1:]
                )["representation_role"].strip('"'),
                "represented_value": dict(
                    line.split("=", 1) for line in sections[name].splitlines()[1:]
                )["represented_value"].strip('"'),
                **added[name],
                "block_only_harmony_diagnostic_only": True,
                "block_only_harmony_warning": "Diagnostic only: block-only Harmony counts include their own Harmony conversation envelope and must not reconstruct complete-prompt counts.",
            })
    return prompt_records, rendered_records, block_records


def _delta(informative: dict, reference: dict, field: str) -> int:
    return informative[field] - reference[field]


def primary_contrast_deltas(prompt_records, block_records) -> dict[str, Any]:
    prompts = {(record["case_id"], record["condition"]): record for record in prompt_records}
    blocks = {
        (record["case_id"], record["condition"], record["block_title"]): record
        for record in block_records
    }
    result = {}
    for hypothesis, contrast in CONTRASTS.items():
        case_results = {}
        for case_id in contrast["cases"]:
            informative = prompts[(case_id, contrast["informative"])]
            reference = prompts[(case_id, contrast["reference"])]
            informative_block = blocks[
                (case_id, contrast["informative"], contrast["informative_block"])
            ]
            reference_block = blocks[
                (case_id, contrast["reference"], contrast["reference_block"])
            ]
            case_results[case_id] = {
                "formula": f'{contrast["informative"]} minus {contrast["reference"]}',
                "raw_full_prompt_token_delta": _delta(informative, reference, "raw_token_count"),
                "harmony_full_prompt_token_delta": _delta(informative, reference, "harmony_token_count"),
                "character_delta": _delta(informative, reference, "character_count"),
                "utf8_byte_delta": _delta(informative, reference, "utf8_byte_count"),
                "line_delta": _delta(informative, reference, "line_count"),
                "block_only_raw_token_delta": _delta(informative_block, reference_block, "raw_token_count"),
                "block_only_harmony_token_delta": _delta(informative_block, reference_block, "harmony_token_count"),
            }
        result[hypothesis] = case_results
    return result


def interaction_diagnostics(prompt_records) -> dict[str, Any]:
    prompts = {(record["case_id"], record["condition"]): record for record in prompt_records}
    result = {}
    fields = {
        "raw_tokens": "raw_token_count",
        "harmony_tokens": "harmony_token_count",
        "characters": "character_count",
        "utf8_bytes": "utf8_byte_count",
    }
    for hypothesis, contrast in CONTRASTS.items():
        first_case, second_case = contrast["cases"]
        measurements = {}
        for label, field in fields.items():
            informative_pair = (
                prompts[(first_case, contrast["informative"])][field]
                - prompts[(second_case, contrast["informative"])][field]
            )
            reference_pair = (
                prompts[(first_case, contrast["reference"])][field]
                - prompts[(second_case, contrast["reference"])][field]
            )
            measurements[label] = {
                "informative_first_case_minus_second_case": informative_pair,
                "reference_first_case_minus_second_case": reference_pair,
                "difference_of_pairwise_differences": informative_pair - reference_pair,
            }
        result[hypothesis] = {
            "informative_condition": contrast["informative"],
            "reference_condition": contrast["reference"],
            "first_case": first_case,
            "second_case": second_case,
            "measurements": measurements,
            "scientific_hypothesis_test": False,
        }
    return result


def acceptance_vector(primary_deltas: dict[str, Any], interactions: dict[str, Any]) -> list[int]:
    primary_token_values = [
        case_result[field]
        for hypothesis in primary_deltas.values()
        for case_result in hypothesis.values()
        for field in (
            "raw_full_prompt_token_delta",
            "harmony_full_prompt_token_delta",
        )
    ]
    interaction_token_values = [
        hypothesis["measurements"][measure]["difference_of_pairwise_differences"]
        for hypothesis in interactions.values()
        for measure in ("raw_tokens", "harmony_tokens")
    ]
    return primary_token_values + interaction_token_values


def classify(primary_deltas: dict[str, Any], interactions: dict[str, Any]) -> str:
    return (
        "exact_primary_token_parity"
        if acceptance_vector(primary_deltas, interactions) == [0] * 12
        else "nonzero_primary_token_difference"
    )


def exit_code_for_classification(classification: str) -> int:
    return 1 if classification == "runtime_failure" else 0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records) -> None:
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _initialize(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    placeholders = {
        "content_matched_tokenizer_summary.json": {
            "classification": "runtime_failure",
            "real_execution_authorized": False,
            "model_or_api_execution": False,
        },
        "primary_contrast_token_deltas.json": {},
        "interaction_token_diagnostics.json": {},
        "v0_4_0_audit_snapshot.json": {},
        "dependency_versions.json": {"status": "not_loaded"},
        "immutable_input_hashes.json": {"status": "not_verified"},
        "execution_environment.json": {
            "python": sys.version,
            "platform": platform.platform(),
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "model_or_api_execution": False,
        },
    }
    for filename, value in placeholders.items():
        _write_json(output_dir / filename, value)
    for filename in (
        "prompt_token_records.jsonl",
        "rendered_prompts.jsonl",
        "block_token_records.jsonl",
    ):
        (output_dir / filename).write_text("")


def execute(output_dir: Path) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    _initialize(output_dir, manifest)
    try:
        immutable = verify_immutable_inputs()
        _write_json(output_dir / "immutable_input_hashes.json", immutable)
        audit = run_audit()
        audit_snapshot = {
            key: value
            for key, value in audit.items()
        }
        _write_json(output_dir / "v0_4_0_audit_snapshot.json", audit_snapshot)
        failed_audits = [
            key for key, expected in STRUCTURAL_AUDIT_REQUIREMENTS.items()
            if audit.get(key) != expected
        ]
        if len(audit.get("pairwise_audits", ())) != 6 or not all(
            pair.get("passed") for pair in audit.get("pairwise_audits", ())
        ):
            failed_audits.append("pairwise_audits")
        if failed_audits:
            raise RuntimeError("v0.4.0 structural audit failed: " + ", ".join(failed_audits))
        prompts = render_all_prompts()
        if not (
            len(prompts) == 16
            and sum(prompt.family == "DP5" for prompt in prompts) == 6
            and sum(prompt.family == "DP7" for prompt in prompts) == 10
        ):
            raise RuntimeError("frozen prompt population mismatch")
        raw_encoding, harmony_encode, versions = load_tokenizers(manifest)
        _write_json(output_dir / "dependency_versions.json", versions)
        prompt_records, rendered_records, block_records = measure_prompts(
            prompts, raw_encoding.encode, harmony_encode
        )
        _write_jsonl(output_dir / "prompt_token_records.jsonl", prompt_records)
        _write_jsonl(output_dir / "rendered_prompts.jsonl", rendered_records)
        _write_jsonl(output_dir / "block_token_records.jsonl", block_records)
        primary = primary_contrast_deltas(prompt_records, block_records)
        interactions = interaction_diagnostics(prompt_records)
        _write_json(output_dir / "primary_contrast_token_deltas.json", primary)
        _write_json(output_dir / "interaction_token_diagnostics.json", interactions)
        classification = classify(primary, interactions)
        exact_parity = classification == "exact_primary_token_parity"
        _write_json(output_dir / "content_matched_tokenizer_summary.json", {
            "classification": classification,
            "exact_parity": exact_parity,
            "primary_contrast_token_deltas": primary,
            "interaction_token_diagnostics": interactions,
            "prompt_count": len(prompt_records),
            "prompt_counts": {
                "DP5": sum(record["family"] == "DP5" for record in prompt_records),
                "DP7": sum(record["family"] == "DP7" for record in prompt_records),
                "total": len(prompt_records),
            },
            "dependency_versions": versions,
            "scientific_inference": None,
            "nonzero_difference_interpretation": (
                None if exact_parity
                else "Measured token difference only; not scientific evidence and not subject to a tolerance threshold."
            ),
            "real_execution_authorized": False,
            "model_or_api_execution": False,
        })
        return exit_code_for_classification(classification)
    except Exception as error:
        _write_json(output_dir / "content_matched_tokenizer_summary.json", {
            "classification": "runtime_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "scientific_inference": None,
            "real_execution_authorized": False,
            "model_or_api_execution": False,
        })
        return exit_code_for_classification("runtime_failure")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return execute(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
