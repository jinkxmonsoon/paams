"""Exhaustive tokenizer-only selection over the frozen v0.3.2 wording bank."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from .decision_point_natural_control_prompting_v0_3_0 import render_prompt as render_parent_prompt
from .decision_point_natural_wording_bank_v0_3_2 import (
    EXPECTED_FIELDS_BY_SECTION,
    FUTURE_SELECTION_RULE,
    PERMITTED_REPLACEMENT_FIELDS,
    audit_prompt_structure,
    audit_wording_bank,
    candidate_specifications,
    render_candidate_set,
)
from .decision_point_scenarios import CASES


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_natural_wording_selection_manifest_v0_3_3.json"
)
CALIBRATION_VERSION = "btom-v2-decision-point-natural-wording-selection-0.3.3"
EXPECTED_CANDIDATES = 1296
EXPECTED_PROMPTS_PER_CANDIDATE = 16
IMMUTABLE_PARENT_SHA256 = {
    "btom_v2/decision_point_natural_wording_bank_manifest_v0_3_2.json":
        "b45ef198b4ab8ab003a47247301834a0321e77ed87f0ee1ce12da078dd825cc5",
    "btom_v2/decision_point_natural_wording_bank_v0_3_2.py":
        "48f3d2a45a971eb5610e03ea3bac42c086d555783dda3762e1db597b45072373",
    "btom_v2/decision_point_natural_wording_bank_design_v0_3_2.md":
        "d112ea743d9d413c64d575261f89915cf8a813f17d4ec262302821b2dc5516be",
    "tests/test_decision_point_natural_wording_bank_v0_3_2.py":
        "3a66db3e6ea208bdcc0e3dfd7fc4ff37a2c4bd90fd3d777009934c11a05deb0d",
    "btom_v2/decision_point_natural_control_prompting_v0_3_0.py":
        "08df09f8e5d950f0fde422993133ca59e40362bcd06ba799446951a5701c1961",
    "btom_v2/decision_point_scenarios.py":
        "e7a936cca701356dc2f9a5c8ab9621b5e593af932b854afe4c9a146acf1994f5",
}
PRIMARY = {
    "H1_DP5a": ("DP5a_self_belief_false", "explicit_first_order", "operational_metadata_reference"),
    "H1_DP5b": ("DP5b_self_belief_current", "explicit_first_order", "operational_metadata_reference"),
    "H2_DP7a": ("DP7a_partner_belief_stale", "explicit_second_order", "first_order_plus_message_provenance"),
    "H2_DP7b": ("DP7b_partner_belief_current", "explicit_second_order", "first_order_plus_message_provenance"),
}
INTERACTIONS = {
    "H1": (
        ("DP5a_self_belief_false", "explicit_first_order"),
        ("DP5b_self_belief_current", "explicit_first_order"),
        ("DP5a_self_belief_false", "operational_metadata_reference"),
        ("DP5b_self_belief_current", "operational_metadata_reference"),
    ),
    "H2": (
        ("DP7a_partner_belief_stale", "explicit_second_order"),
        ("DP7b_partner_belief_current", "explicit_second_order"),
        ("DP7a_partner_belief_stale", "first_order_plus_message_provenance"),
        ("DP7b_partner_belief_current", "first_order_plus_message_provenance"),
    ),
}
STRUCTURAL_FLAGS = (
    "section_order_frozen", "field_names_frozen", "field_order_frozen",
    "field_counts_frozen", "line_counts_frozen",
    "permitted_replacement_boundary_passed",
)
NO_SCIENTIFIC_INFERENCE = (
    "Technical tokenizer feasibility only; no scientific hypothesis is tested."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_immutable_inputs() -> dict[str, Any]:
    records = {}
    failures = []
    for relative, expected in IMMUTABLE_PARENT_SHA256.items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
        if actual != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError("immutable input verification failed: " + ", ".join(failures))
    return records


def verify_bank_population() -> tuple[tuple, dict[str, Any]]:
    specifications = candidate_specifications()
    expected_schema = (
        "location_surface_id", "h1_operational_variant_id",
        "h2_provenance_variant_id", "candidate_id",
    )
    if len(specifications) != EXPECTED_CANDIDATES:
        raise RuntimeError("frozen candidate count mismatch")
    if len({item.candidate_id for item in specifications}) != EXPECTED_CANDIDATES:
        raise RuntimeError("frozen candidate IDs are not unique")
    if tuple(field.name for field in fields(type(specifications[0]))) != expected_schema:
        raise RuntimeError("candidate specification schema is not the frozen lowercase schema")
    ordering = tuple(
        (item.location_surface_id, item.h1_operational_variant_id, item.h2_provenance_variant_id)
        for item in specifications
    )
    if ordering != tuple(sorted(ordering)):
        raise RuntimeError("frozen candidate ordering mismatch")
    audit = audit_wording_bank()
    required = {
        "candidate_sets_audited": 1296,
        "prompts_audited": 20736,
        "candidate_sets_failed": 0,
        "section_order_frozen": True,
        "field_names_frozen": True,
        "field_order_frozen": True,
        "field_counts_frozen": True,
        "line_counts_frozen": True,
        "permitted_replacement_boundary_passed": True,
        "field_and_line_counts_frozen": True,
    }
    failures = [key for key, value in required.items() if audit.get(key) != value]
    if failures or audit.get("failed_candidate_ids") != []:
        raise RuntimeError("frozen bank audit failed: " + ", ".join(failures))
    if FUTURE_SELECTION_RULE["tolerance_threshold"] is not None:
        raise RuntimeError("frozen no-tolerance rule changed")
    if len(PERMITTED_REPLACEMENT_FIELDS) != 7 or len(EXPECTED_FIELDS_BY_SECTION) != 4:
        raise RuntimeError("frozen structural schema changed")
    return specifications, audit


def load_tokenizers(manifest: dict[str, Any]):
    # Third-party tokenizer imports are runtime-only and occur after parent checks.
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
        raise RuntimeError(f"dependency version mismatch: {versions}")
    raw_encoding = tiktoken.get_encoding("o200k_harmony")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def harmony_tokens(prompt: str) -> list[int]:
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.USER, prompt)
        ])
        return list(harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        ))

    return raw_encoding.encode, harmony_tokens, versions


def _token_hash(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def measure_prompt(prompt, raw_encode: Callable, harmony_encode: Callable) -> dict[str, Any]:
    raw_ids = list(raw_encode(prompt.prompt))
    harmony_ids = list(harmony_encode(prompt.prompt))
    return {
        "prompt_id": prompt.prompt_id,
        "case_id": prompt.case_id,
        "family": prompt.family,
        "condition": prompt.condition,
        "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
        "character_count": len(prompt.prompt),
        "raw_token_count": len(raw_ids),
        "raw_token_id_sha256": _token_hash(raw_ids),
        "harmony_token_count": len(harmony_ids),
        "harmony_token_id_sha256": _token_hash(harmony_ids),
        "model_or_api_execution": False,
    }


def calculate_token_diagnostics(counts: dict[tuple[str, str], dict[str, int]]) -> dict[str, Any]:
    primary_deltas = {}
    for label, (case_id, informative, reference) in PRIMARY.items():
        primary_deltas[label] = {
            token_kind: counts[(case_id, informative)][token_kind] - counts[(case_id, reference)][token_kind]
            for token_kind in ("raw_token_count", "harmony_token_count")
        }
    interactions = {}
    for label, (inf_a, inf_b, ref_a, ref_b) in INTERACTIONS.items():
        interactions[label] = {
            token_kind: (
                counts[inf_a][token_kind] - counts[inf_b][token_kind]
                - (counts[ref_a][token_kind] - counts[ref_b][token_kind])
            )
            for token_kind in ("raw_token_count", "harmony_token_count")
        }
    acceptance_vector = [
        primary_deltas[label][token_kind]
        for label in ("H1_DP5a", "H1_DP5b", "H2_DP7a", "H2_DP7b")
        for token_kind in ("raw_token_count", "harmony_token_count")
    ] + [
        interactions[label][token_kind]
        for label in ("H1", "H2")
        for token_kind in ("raw_token_count", "harmony_token_count")
    ]
    return {
        "primary_deltas": primary_deltas,
        "interaction_diagnostics": interactions,
        "acceptance_vector": acceptance_vector,
        "all_token_deltas_exactly_zero": acceptance_vector == [0] * 12,
    }


def _block_fields(block: str) -> dict[str, str]:
    return dict(line.split("=", 1)[0:2] for line in block.splitlines()[1:])


def calculate_ranking_metrics(prompts, specification) -> dict[str, Any]:
    by_key = {(prompt.case_id, prompt.condition): prompt for prompt in prompts}
    blocks = (
        ("DP5a_self_belief_false", "explicit_first_order", "FIRST-ORDER REPRESENTATION"),
        ("DP5a_self_belief_false", "operational_metadata_reference", "DECISION METADATA"),
        ("DP5b_self_belief_current", "explicit_first_order", "FIRST-ORDER REPRESENTATION"),
        ("DP5b_self_belief_current", "operational_metadata_reference", "DECISION METADATA"),
        ("DP7a_partner_belief_stale", "explicit_second_order", "SECOND-ORDER REPRESENTATION"),
        ("DP7a_partner_belief_stale", "first_order_plus_message_provenance", "MESSAGE PROVENANCE METADATA"),
        ("DP7b_partner_belief_current", "explicit_second_order", "SECOND-ORDER REPRESENTATION"),
        ("DP7b_partner_belief_current", "first_order_plus_message_provenance", "MESSAGE PROVENANCE METADATA"),
    )
    total_characters = sum(
        len(dict(by_key[(case_id, condition)].sections)[block])
        for case_id, condition, block in blocks
    )
    h1_fields = _block_fields(dict(by_key[(blocks[1][0], blocks[1][1])].sections)[blocks[1][2]])
    h2_fields = _block_fields(dict(by_key[(blocks[5][0], blocks[5][1])].sections)[blocks[5][2]])
    changed_literals = 2
    changed_literals += sum(
        h1_fields[field] != f'"{baseline}"'
        for field, baseline in (
            ("decision_scope", "single_action"),
            ("action_source", "listed_actions"),
            ("output_mode", "json_object"),
        )
    )
    changed_literals += sum(
        h2_fields[field] != f'"{baseline}"'
        for field, baseline in (
            ("message_type", "structured_statement"),
            ("delivery_status", "available"),
        )
    )
    specification_dict = {
        "h1_operational_variant_id": specification.h1_operational_variant_id,
        "h2_provenance_variant_id": specification.h2_provenance_variant_id,
        "location_surface_id": specification.location_surface_id,
    }
    canonical = json.dumps(specification_dict, sort_keys=True, separators=(",", ":"))
    return {
        "compared_block_character_count": total_characters,
        "changed_candidate_literal_count": changed_literals,
        "canonical_candidate_specification": canonical,
        "candidate_id_integrity_tiebreaker": specification.candidate_id,
    }


def ranking_key(record: dict[str, Any]) -> tuple:
    metrics = record["ranking_metrics"]
    return (
        metrics["compared_block_character_count"],
        metrics["changed_candidate_literal_count"],
        metrics["canonical_candidate_specification"],
        metrics["candidate_id_integrity_tiebreaker"],
    )


def classify_records(records: list[dict[str, Any]], execution_completed: bool) -> tuple[str, dict | None, int]:
    if not execution_completed or len(records) != EXPECTED_CANDIDATES:
        return "runtime_failure", None, 1
    passing = [record for record in records if record["exact_parity"]]
    if not passing:
        return "no_candidate", None, 0
    return "success", min(passing, key=ranking_key), 0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _write_jsonl(path: Path, values) -> None:
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":"), default=str) + "\n")


def execute(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    immutable_records = verify_immutable_inputs()
    manifest = json.loads(MANIFEST_PATH.read_text())
    specifications, bank_audit = verify_bank_population()
    _write_json(output_dir / "immutable_input_hashes.json", immutable_records)
    _write_json(output_dir / "bank_audit_snapshot.json", bank_audit)
    raw_encode, harmony_encode, versions = load_tokenizers(manifest)
    _write_json(output_dir / "dependency_versions.json", versions)
    cases_by_id = {case.case_id: case for case in CASES}
    diagnostics = []
    structurally_failed = []

    for ordinal, specification in enumerate(specifications, 1):
        prompts = render_candidate_set(specification)
        if len(prompts) != EXPECTED_PROMPTS_PER_CANDIDATE:
            raise RuntimeError("candidate prompt count mismatch")
        prompt_records = []
        structural_passed = True
        counts = {}
        for prompt in prompts:
            case = cases_by_id[prompt.case_id]
            parent = render_parent_prompt(case, prompt.condition)
            structure = audit_prompt_structure(prompt, parent, case)
            structural_passed &= structure["passed"]
            measurement = measure_prompt(prompt, raw_encode, harmony_encode)
            measurement["structural_audit"] = structure
            prompt_records.append(measurement)
            counts[(prompt.case_id, prompt.condition)] = measurement
        token_diagnostics = calculate_token_diagnostics(counts)
        ranking_metrics = calculate_ranking_metrics(prompts, specification)
        exact_parity = structural_passed and token_diagnostics["all_token_deltas_exactly_zero"]
        record = {
            "ordinal": ordinal,
            "candidate_id": specification.candidate_id,
            "specification": {
                "location_surface_id": specification.location_surface_id,
                "h1_operational_variant_id": specification.h1_operational_variant_id,
                "h2_provenance_variant_id": specification.h2_provenance_variant_id,
            },
            "structural_audit_passed": structural_passed,
            **token_diagnostics,
            "exact_parity": exact_parity,
            "ranking_metrics": ranking_metrics,
            "prompt_token_records": prompt_records,
            "no_scientific_inference": NO_SCIENTIFIC_INFERENCE,
        }
        diagnostics.append(record)
        if not structural_passed:
            structurally_failed.append(specification.candidate_id)
    classification, selected, exit_code = classify_records(diagnostics, execution_completed=True)
    passing = sorted((record for record in diagnostics if record["exact_parity"]), key=ranking_key)
    selected_payload = {
        "candidate_id": selected["candidate_id"] if selected else None,
        "specification": selected["specification"] if selected else None,
        "rank": 1 if selected else None,
        "ranking_metrics": selected["ranking_metrics"] if selected else None,
        "classification": classification,
        "real_execution_authorized": False,
    }
    completion = {
        "expected_candidates": EXPECTED_CANDIDATES,
        "evaluated_candidates": len(diagnostics),
        "skipped_candidates": EXPECTED_CANDIDATES - len(diagnostics),
        "duplicate_ids": len(diagnostics) - len({record["candidate_id"] for record in diagnostics}),
        "structurally_failed_candidates": structurally_failed,
        "exact_parity_candidate_count": len(passing),
        "execution_completed": len(diagnostics) == EXPECTED_CANDIDATES,
    }
    summary = {
        "classification": classification,
        "calibration_version": CALIBRATION_VERSION,
        "technical_hypothesis": "T-CAL-1",
        "selected_candidate_id": selected_payload["candidate_id"],
        "exact_parity_candidate_count": len(passing),
        "search_completion": completion,
        "dependency_versions": versions,
        "no_scientific_inference": NO_SCIENTIFIC_INFERENCE,
        "model_or_api_execution": False,
        "real_execution_authorized": False,
    }
    _write_json(output_dir / "wording_selection_summary.json", summary)
    _write_json(output_dir / "selected_candidate.json", selected_payload)
    _write_jsonl(output_dir / "exact_parity_candidates.jsonl", passing)
    _write_jsonl(output_dir / "candidate_token_diagnostics.jsonl", diagnostics)
    _write_json(output_dir / "search_completion_diagnostics.json", completion)
    selected_id = selected_payload["candidate_id"]
    selected_prompt_records = []
    selected_rendered_prompts = []
    if selected_id:
        selected_index = next(
            index for index, specification in enumerate(specifications)
            if specification.candidate_id == selected_id
        )
        selected_prompts = render_candidate_set(specifications[selected_index])
        selected_prompt_records = selected["prompt_token_records"]
        selected_rendered_prompts = [
            {
                "prompt_id": prompt.prompt_id,
                "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt.prompt,
            }
            for prompt in selected_prompts
        ]
    _write_jsonl(
        output_dir / "selected_candidate_prompt_records.jsonl",
        selected_prompt_records,
    )
    _write_jsonl(
        output_dir / "selected_candidate_rendered_prompts.jsonl",
        selected_rendered_prompts,
    )
    _write_json(output_dir / "execution_environment.json", {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_or_api_execution": False,
    })
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return execute(args.output_dir)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "wording_selection_summary.json", {
            "classification": "runtime_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "model_or_api_execution": False,
            "real_execution_authorized": False,
        })
        _write_json(args.output_dir / "selected_candidate.json", {
            "candidate_id": None, "specification": None, "rank": None,
            "ranking_metrics": None, "classification": "runtime_failure",
            "real_execution_authorized": False,
        })
        _write_jsonl(args.output_dir / "exact_parity_candidates.jsonl", [])
        _write_jsonl(args.output_dir / "candidate_token_diagnostics.jsonl", [])
        _write_jsonl(args.output_dir / "selected_candidate_prompt_records.jsonl", [])
        _write_jsonl(args.output_dir / "selected_candidate_rendered_prompts.jsonl", [])
        _write_json(args.output_dir / "bank_audit_snapshot.json", None)
        _write_json(args.output_dir / "search_completion_diagnostics.json", {
            "expected_candidates": EXPECTED_CANDIDATES,
            "evaluated_candidates": 0,
            "skipped_candidates": EXPECTED_CANDIDATES,
            "duplicate_ids": 0,
            "structurally_failed_candidates": [],
            "exact_parity_candidate_count": 0,
            "execution_completed": False,
        })
        _write_json(args.output_dir / "dependency_versions.json", None)
        _write_json(args.output_dir / "immutable_input_hashes.json", {
            relative: {
                "expected_sha256": expected,
                "actual_sha256": _sha256(ROOT / relative) if (ROOT / relative).is_file() else None,
            }
            for relative, expected in IMMUTABLE_PARENT_SHA256.items()
        })
        _write_json(args.output_dir / "execution_environment.json", {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_or_api_execution": False,
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
