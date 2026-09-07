"""Tokenizer-only measurement of frozen v0.3.0 natural-control prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .decision_point_natural_control_prompting_v0_3_0 import (
    COMMON_SECTIONS,
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    MESSAGE_PROVENANCE,
    OPERATIONAL_METADATA,
    render_all_prompts,
    run_audit,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_natural_control_tokenizer_manifest_v0_3_1.json"
)
CALIBRATION_VERSION = "btom-v2-decision-point-natural-control-tokenizer-0.3.1"
IMMUTABLE_PARENT_SHA256 = {
    "btom_v2/decision_point_natural_control_manifest_v0_3_0.json":
        "c39b2997282561bdacffa4e6b55e8d26062901e67ce4596a658b569276375549",
    "btom_v2/decision_point_natural_control_prompting_v0_3_0.py":
        "08df09f8e5d950f0fde422993133ca59e40362bcd06ba799446951a5701c1961",
    "btom_v2/decision_point_natural_control_design_v0_3_0.md":
        "c4c3fbc1de8c4e13af89d56cc5077ab88edac95e7e1edec0f54391a88180d8e4",
    "btom_v2/decision_point_prompting_v0_2_1.py":
        "a2a6140e4e7676187c589a7e10c61fc6ff1b96f5aa0ba90258d54bb265e8b80f",
    "btom_v2/decision_point_prompt_protocol_v0_2_1.json":
        "9acec8dba50f0e33082cb8b0b238b8f4d27875b6ee4382fdee7c7e4ea806e8f7",
    "btom_v2/decision_point_scenarios.py":
        "e7a936cca701356dc2f9a5c8ab9621b5e593af932b854afe4c9a146acf1994f5",
}
STRUCTURAL_AUDIT_REQUIREMENTS = {
    "prompt_count": 16,
    "DP5_prompt_count": 6,
    "DP7_prompt_count": 10,
    "common_section_immutability_passed": True,
    "valid_actions_frozen": True,
    "raw_messages_frozen": True,
    "DP7_raw_message_exactly_once": True,
    "hidden_field_leak_count": 0,
    "scoring_label_leak_count": 0,
    "condition_name_leak_count": 0,
    "source_message_field_count": 0,
    "legacy_filler_count": 0,
    "H1_equal_field_and_line_counts": True,
    "H2_equal_field_and_line_counts": True,
    "tokenizer_parity_claimed": False,
    "model_or_API_execution": False,
    "real_execution_authorized": False,
}
CONTRASTS = {
    "H1": {
        "informative": EXPLICIT_FIRST,
        "reference": OPERATIONAL_METADATA,
        "cases": ("DP5a_self_belief_false", "DP5b_self_belief_current"),
        "informative_block": "FIRST-ORDER REPRESENTATION",
        "reference_block": "DECISION METADATA",
    },
    "H2": {
        "informative": EXPLICIT_SECOND,
        "reference": MESSAGE_PROVENANCE,
        "cases": ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"),
        "informative_block": "SECOND-ORDER REPRESENTATION",
        "reference_block": "MESSAGE PROVENANCE METADATA",
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
        common = {
            name: _text_measurement(sections[name], raw_encode, harmony_encode)
            for name in COMMON_SECTIONS
        }
        added_names = tuple(name for name, _ in prompt.sections if name not in COMMON_SECTIONS)
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
            "common_section_token_counts": {
                name: {
                    "raw": measurement["raw_token_count"],
                    "harmony": measurement["harmony_token_count"],
                }
                for name, measurement in common.items()
            },
            "added_block_token_counts": {
                name: {
                    "raw": measurement["raw_token_count"],
                    "harmony": measurement["harmony_token_count"],
                }
                for name, measurement in added.items()
            },
            "included_block_names": added_names,
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
                "block_name": name,
                "block_text": sections[name],
                **added[name],
                "block_only_counts_are_full_prompt_equivalent": False,
            })
    return prompt_records, rendered_records, block_records


def _delta(informative: dict, reference: dict, field: str) -> int:
    return informative[field] - reference[field]


def primary_contrast_deltas(prompt_records, block_records) -> dict[str, Any]:
    prompts = {(record["case_id"], record["condition"]): record for record in prompt_records}
    blocks = {
        (record["case_id"], record["condition"], record["block_name"]): record
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


def classify(primary_deltas: dict[str, Any], interactions: dict[str, Any]) -> str:
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
    return (
        "exact_primary_token_parity"
        if all(value == 0 for value in primary_token_values + interaction_token_values)
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
        "natural_control_tokenizer_summary.json": {
            "classification": "runtime_failure",
            "real_execution_authorized": False,
            "model_or_api_execution": False,
        },
        "primary_contrast_token_deltas.json": {},
        "interaction_token_diagnostics.json": {},
        "v0_3_0_audit_snapshot.json": {},
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
            key: ([asdict(record) for record in value] if key == "prompt_records" else value)
            for key, value in audit.items()
        }
        _write_json(output_dir / "v0_3_0_audit_snapshot.json", audit_snapshot)
        failed_audits = [
            key for key, expected in STRUCTURAL_AUDIT_REQUIREMENTS.items()
            if audit.get(key) != expected
        ]
        if failed_audits:
            raise RuntimeError("v0.3.0 structural audit failed: " + ", ".join(failed_audits))
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
        _write_json(output_dir / "natural_control_tokenizer_summary.json", {
            "classification": classification,
            "exact_parity": exact_parity,
            "primary_contrast_token_deltas": primary,
            "interaction_token_diagnostics": interactions,
            "prompt_count": len(prompt_records),
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
        _write_json(output_dir / "natural_control_tokenizer_summary.json", {
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
