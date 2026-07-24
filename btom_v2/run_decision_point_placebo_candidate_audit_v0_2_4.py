"""Tokenizer-only audit of typed placebo location-code candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from .decision_point_prompting import (
    EXPLICIT_FIRST, EXPLICIT_SECOND, HIDDEN_FIELD_NAMES, NEUTRAL_FIRST,
    NEUTRAL_SECOND, REACTIVE, forbidden_label_count,
)
from .decision_point_prompting_v0_2_1 import render_all_prompts as render_parent_prompts
from .decision_point_scenarios import CASE_BY_ID
from .run_decision_point_first_order_feasibility_v0_2_3 import (
    ADDITIONAL_IMMUTABLE_INPUTS as V023_PARENT_INPUTS,
    _neutral_first, _neutral_second, _parents, _token_counts,
)
from .run_decision_point_tokenizer_calibration_v0_2_2 import (
    IMMUTABLE_INPUTS as V021_INPUTS, _block_shape, load_tokenizers,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_placebo_candidate_manifest_v0_2_4.json"
)
CALIBRATION_VERSION = "btom-v2-decision-point-placebo-candidate-audit-0.2.4"
PREFIX = "loc_"
PAYLOAD_ALPHABET = "DFGHJKLMNPQRTVWXYZ23456789"
PANEL_SIZE = 20
SEARCH_BOUNDS = {
    "DP5_first_order": 500000,
    "DP7_first_order": 500000,
    "DP7_second_order": 500000,
}
PARENT_RUN_ID = 30066731354
PARENT_ARTIFACT_DIGEST = "0294867f039827410c9e245324f52fe8ed19b80a8adf813e3f3ed5baf440d72b"
LEGACY_SECOND_ORDER_VARIANT_INDEX = 3087
ADDITIONAL_IMMUTABLE_INPUTS = {
    "btom_v2/decision_point_first_order_feasibility_manifest_v0_2_3.json":
        "0791b2b318577d3f312f3da147facaefb5819f13b5d54d4c3ba599557f10e480",
    "btom_v2/run_decision_point_first_order_feasibility_v0_2_3.py":
        "aa563de6f61ae93e36b02641104106253d345a498be7830b48a67e3298798bd8",
}
TASK_TERMS = (
    "agent", "belief", "box", "control", "corridor", "correction", "current",
    "decoy", "false", "kit", "medical", "move", "neutral", "oracle",
    "productive", "rescue", "room", "score", "send", "staging", "stale",
    "target", "treatment", "true", "victim",
)
EXPECTED_CHANGED = {
    "DP5a_self_belief_false:reactive_neutral_first_order_matched",
    "DP5b_self_belief_current:reactive_neutral_first_order_matched",
    "DP7a_partner_belief_stale:reactive_neutral_first_order_matched",
    "DP7b_partner_belief_current:reactive_neutral_first_order_matched",
    "DP7a_partner_belief_stale:first_order_neutral_second_order_matched",
    "DP7b_partner_belief_current:first_order_neutral_second_order_matched",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_immutable_inputs() -> dict[str, Any]:
    expected = {**V021_INPUTS, **V023_PARENT_INPUTS, **ADDITIONAL_IMMUTABLE_INPUTS}
    records = {}
    failures = []
    for relative, digest in expected.items():
        actual = _sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        matches = actual == digest
        records[relative] = {"expected_sha256": digest, "actual_sha256": actual, "matches": matches}
        if not matches:
            failures.append(relative)
    parent = json.loads(
        (ROOT / "btom_v2/decision_point_prompt_protocol_v0_2_1.json").read_text()
    )
    records["parent_protocol_authorization_valid"] = parent.get("real_execution_authorized") is False
    if failures or not records["parent_protocol_authorization_valid"]:
        raise RuntimeError("immutable input verification failed: " + ", ".join(failures))
    return records


def payload_metrics(payload: str) -> dict[str, int]:
    runs = [len(match.group(0)) for match in re.finditer(r"(.)\1*", payload)]
    bigrams = [payload[index:index + 2] for index in range(len(payload) - 1)]
    return {
        "distinct_payload_character_count": len(set(payload)),
        "maximum_repeated_character_run": max(runs, default=0),
        "repeated_bigram_count": len(bigrams) - len(set(bigrams)),
    }


def _periodic(payload: str, period: int) -> bool:
    return all(character == payload[index % period] for index, character in enumerate(payload))


def anti_salience_requirements(value: str) -> dict[str, bool]:
    payload = value[len(PREFIX):] if value.startswith(PREFIX) else ""
    metrics = payload_metrics(payload)
    lowered = value.lower()
    return {
        "typed_prefix": value.startswith(PREFIX),
        "valid_total_length": len(value) in {8, 10},
        "payload_alphabet": bool(payload) and set(payload) <= set(PAYLOAD_ALPHABET),
        "minimum_three_distinct": metrics["distinct_payload_character_count"] >= 3,
        "maximum_run_two": metrics["maximum_repeated_character_run"] <= 2,
        "no_repeated_bigram": metrics["repeated_bigram_count"] == 0,
        "not_period_one": not _periodic(payload, 1),
        "not_period_two": not _periodic(payload, 2),
        "no_agent_letters_in_payload": not any(letter in payload for letter in "ABC"),
        "no_task_semantics": not any(term in lowered for term in TASK_TERMS),
        "no_legacy_repeated_placeholder": "BBBBBBBB" not in value,
        "no_legacy_second_order_placeholder": "989_____" not in value,
        "underscore_only_in_prefix": "_" not in payload and value.count("_") == 1,
        "no_whitespace": not any(character.isspace() for character in value),
        "ascii_only": value.isascii(),
    }


def typed_placebo_candidates(payload_length: int) -> Iterator[str]:
    """Enumerate typed placebo codes without case or condition identifiers."""
    for symbols in itertools.product(PAYLOAD_ALPHABET, repeat=payload_length):
        value = PREFIX + "".join(symbols)
        if all(anti_salience_requirements(value).values()):
            yield value


def _token_hash(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _candidate_id(search_name: str, short_value: str, long_value: str | None) -> str:
    specification = {
        "generator_family": "typed_placebo_location_code",
        "long_value": long_value,
        "parent_artifact_digest": PARENT_ARTIFACT_DIGEST,
        "parent_run_id": PARENT_RUN_ID,
        "search_name": search_name,
        "short_value": short_value,
    }
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _prompt_token_stats(prompt, raw_encode: Callable, harmony_encode: Callable) -> dict[str, Any]:
    raw_ids = list(raw_encode(prompt.prompt))
    harmony_ids = list(harmony_encode(prompt.prompt))
    return {
        "raw_token_count": len(raw_ids),
        "harmony_token_count": len(harmony_ids),
        "raw_token_id_sha256": _token_hash(raw_ids),
        "harmony_token_id_sha256": _token_hash(harmony_ids),
    }


def _evaluate_pair(
    search_name: str,
    long_value: str,
    raw_encode: Callable,
    harmony_encode: Callable,
) -> dict[str, Any]:
    parents = _parents()
    if search_name == "DP5_first_order":
        pair = ("DP5a_self_belief_false", "DP5b_self_belief_current")
        conditions = (NEUTRAL_FIRST, EXPLICIT_FIRST)
        replace = _neutral_first
    elif search_name == "DP7_second_order":
        pair = ("DP7a_partner_belief_stale", "DP7b_partner_belief_current")
        conditions = (NEUTRAL_SECOND, EXPLICIT_SECOND)
        replace = _neutral_second
    else:
        raise ValueError("unsupported pair search")
    values = {pair[0]: long_value, pair[1]: long_value[:8]}
    details = {}
    requirements = {"prefix_coupled": long_value.startswith(long_value[:8])}
    for label in pair:
        neutral = replace(parents[f"{label}:{conditions[0]}"], values[label])
        explicit = parents[f"{label}:{conditions[1]}"]
        neutral_stats = _prompt_token_stats(neutral, raw_encode, harmony_encode)
        explicit_stats = _prompt_token_stats(explicit, raw_encode, harmony_encode)
        neutral_counts = (neutral_stats["raw_token_count"], neutral_stats["harmony_token_count"])
        explicit_counts = (explicit_stats["raw_token_count"], explicit_stats["harmony_token_count"])
        requirements[f"{label}_structure"] = (
            _block_shape(
                neutral.first_order_block if search_name == "DP5_first_order" else neutral.second_order_block
            )
            == _block_shape(
                explicit.first_order_block if search_name == "DP5_first_order" else explicit.second_order_block
            )
        )
        requirements[f"{label}_raw"] = neutral_counts[0] == explicit_counts[0]
        requirements[f"{label}_harmony"] = neutral_counts[1] == explicit_counts[1]
        if search_name == "DP7_second_order":
            message = CASE_BY_ID[label].raw_delivered_messages[0]
            requirements[f"{label}_evidence_ref"] = 'evidence_ref="raw_message_1"' in neutral.second_order_block
            requirements[f"{label}_raw_message_once"] = neutral.prompt.count(message) == 1
            requirements[f"{label}_source_absent"] = "source_message=" not in neutral.prompt
        details[label] = {"neutral": neutral_stats, "explicit": explicit_stats}
    first, second = pair
    requirements["raw_pair_difference"] = (
        details[first]["neutral"]["raw_token_count"] - details[second]["neutral"]["raw_token_count"]
        == details[first]["explicit"]["raw_token_count"] - details[second]["explicit"]["raw_token_count"]
    )
    requirements["harmony_pair_difference"] = (
        details[first]["neutral"]["harmony_token_count"] - details[second]["neutral"]["harmony_token_count"]
        == details[first]["explicit"]["harmony_token_count"] - details[second]["explicit"]["harmony_token_count"]
    )
    return {"requirements": requirements, "complete_prompt_token_counts": details}


def _evaluate_dp7_first(value: str, raw_encode: Callable, harmony_encode: Callable) -> dict[str, Any]:
    parents = _parents()
    pair = ("DP7a_partner_belief_stale", "DP7b_partner_belief_current")
    details = {}
    requirements = {"same_value_across_pair": True}
    for label in pair:
        neutral = _neutral_first(parents[f"{label}:{NEUTRAL_FIRST}"], value)
        explicit = parents[f"{label}:{EXPLICIT_FIRST}"]
        neutral_stats = _prompt_token_stats(neutral, raw_encode, harmony_encode)
        explicit_stats = _prompt_token_stats(explicit, raw_encode, harmony_encode)
        neutral_counts = (neutral_stats["raw_token_count"], neutral_stats["harmony_token_count"])
        explicit_counts = (explicit_stats["raw_token_count"], explicit_stats["harmony_token_count"])
        requirements[f"{label}_structure"] = _block_shape(neutral.first_order_block) == _block_shape(explicit.first_order_block)
        requirements[f"{label}_raw"] = neutral_counts[0] == explicit_counts[0]
        requirements[f"{label}_harmony"] = neutral_counts[1] == explicit_counts[1]
        details[label] = {"neutral": neutral_stats, "explicit": explicit_stats}
    first, second = pair
    requirements["raw_pair_difference"] = details[first]["neutral"]["raw_token_count"] - details[second]["neutral"]["raw_token_count"] == details[first]["explicit"]["raw_token_count"] - details[second]["explicit"]["raw_token_count"]
    requirements["harmony_pair_difference"] = details[first]["neutral"]["harmony_token_count"] - details[second]["neutral"]["harmony_token_count"] == details[first]["explicit"]["harmony_token_count"] - details[second]["explicit"]["harmony_token_count"]
    return {"requirements": requirements, "complete_prompt_token_counts": details}


def _rank_key(record: dict[str, Any]) -> tuple:
    return (
        record["maximum_repeated_character_run"],
        -record["distinct_payload_character_count"],
        record["repeated_bigram_count"],
        record["raw_isolated_value_token_count"],
        record["harmony_isolated_value_token_count"],
        record["candidate_text"],
    )


def _search(
    search_name: str,
    payload_length: int,
    bound: int,
    raw_encode: Callable,
    harmony_encode: Callable,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = 0
    failures: Counter[str] = Counter()
    valid = []
    for candidate in typed_placebo_candidates(payload_length):
        if evaluated >= bound or len(valid) >= PANEL_SIZE:
            break
        evaluated += 1
        short = candidate[:8]
        long_value = candidate if payload_length == 6 else None
        result = (
            _evaluate_pair(search_name, candidate, raw_encode, harmony_encode)
            if payload_length == 6
            else _evaluate_dp7_first(candidate, raw_encode, harmony_encode)
        )
        failed = [name for name, passed in result["requirements"].items() if not passed]
        failures.update(failed)
        if failed:
            continue
        payload = candidate[len(PREFIX):]
        metrics = payload_metrics(payload)
        raw_ids = list(raw_encode(candidate))
        harmony_ids = list(harmony_encode(candidate))
        record = {
            "generator_family": "typed_placebo_location_code",
            "candidate_family": "typed_placebo_location_code",
            "candidate_rank": None,
            "candidate_id": _candidate_id(search_name, short, long_value),
            "search_name": search_name,
            "candidate_text": candidate,
            "short_value": short,
            "long_value": long_value,
            "parent_run_id": PARENT_RUN_ID,
            "parent_artifact_digest": PARENT_ARTIFACT_DIGEST,
            **metrics,
            "raw_isolated_value_token_count": len(raw_ids),
            "harmony_isolated_value_token_count": len(harmony_ids),
            "raw_isolated_token_id_sha256": _token_hash(raw_ids),
            "harmony_isolated_token_id_sha256": _token_hash(harmony_ids),
            "complete_prompt_token_counts": result["complete_prompt_token_counts"],
            "requirement_results": result["requirements"],
        }
        valid.append(record)
    panel = sorted(valid, key=_rank_key)
    for rank, record in enumerate(panel, 1):
        record["candidate_rank"] = rank
    diagnostics = {
        "search_name": search_name,
        "maximum_evaluated_candidates": bound,
        "candidates_evaluated": evaluated,
        "valid_candidates_collected": len(valid),
        "required_panel_size": PANEL_SIZE,
        "failure_counts_by_requirement": dict(sorted(failures.items())),
        "stopped_because": "panel_complete" if len(valid) >= PANEL_SIZE else "bound_exhausted",
    }
    return panel, diagnostics


def _legacy_rejections() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "BBBBBBBB / BBBBBBBBBB",
            "reference_only": True,
            "rejected_for_final_selection": True,
            "rejection_reasons": [
                "repeated-character salience",
                "collision with agent naming convention",
                "first-candidate-only selection",
            ],
        },
        {
            "candidate": "989_____ / 989_____ZZ",
            "legacy_second_order_variant_index": LEGACY_SECOND_ORDER_VARIANT_INDEX,
            "reference_only": True,
            "rejected_for_final_selection": True,
            "rejection_reason": "visually salient untyped placeholder",
        },
    ]


def construct_rank_one_prompts(dp5: dict, dp7_first: dict, dp7_second: dict):
    result = []
    for parent in render_parent_prompts():
        candidate = parent
        if parent.condition == NEUTRAL_FIRST:
            value = dp5["long_value"] if parent.family == "DP5" else dp7_first["short_value"]
            if parent.family == "DP5" and len(CASE_BY_ID[parent.case_id].first_order_representation[0][1]) == 8:
                value = dp5["short_value"]
            candidate = _neutral_first(parent, value)
        elif parent.condition == NEUTRAL_SECOND:
            value = dp7_second["long_value"]
            if len(CASE_BY_ID[parent.case_id].second_order_representation[0].believed_value) == 8:
                value = dp7_second["short_value"]
            candidate = _neutral_second(parent, value)
        result.append(candidate)
    return tuple(result)


def global_audit(prompts, raw_encode: Callable, harmony_encode: Callable) -> dict[str, Any]:
    parents = _parents()
    changed = {prompt.prompt_id for prompt in prompts if prompt.prompt != parents[prompt.prompt_id].prompt}
    parity = {}
    for prompt in prompts:
        if prompt.condition == NEUTRAL_FIRST:
            explicit = parents[f"{prompt.case_id}:{EXPLICIT_FIRST}"]
            parity[prompt.prompt_id] = _token_counts(prompt, raw_encode, harmony_encode) == _token_counts(explicit, raw_encode, harmony_encode)
        elif prompt.condition == NEUTRAL_SECOND:
            explicit = parents[f"{prompt.case_id}:{EXPLICIT_SECOND}"]
            parity[prompt.prompt_id] = _token_counts(prompt, raw_encode, harmony_encode) == _token_counts(explicit, raw_encode, harmony_encode)
    return {
        "prompt_count": len(prompts),
        "changed_prompt_ids": sorted(changed),
        "exactly_six_neutral_prompts_changed": changed == EXPECTED_CHANGED,
        "reactive_prompts_byte_identical": all(prompt.prompt == parents[prompt.prompt_id].prompt for prompt in prompts if prompt.condition == REACTIVE),
        "informative_prompts_byte_identical": all(prompt.prompt == parents[prompt.prompt_id].prompt for prompt in prompts if prompt.condition in {EXPLICIT_FIRST, EXPLICIT_SECOND}),
        "hidden_truth_leak_count": sum(sum(field in prompt.prompt for field in HIDDEN_FIELD_NAMES) for prompt in prompts),
        "forbidden_label_leak_count": sum(forbidden_label_count(prompt.prompt) for prompt in prompts),
        "duplicated_raw_evidence_count": sum(max(0, prompt.prompt.count(message) - 1) for prompt in prompts for message in CASE_BY_ID[prompt.case_id].raw_delivered_messages),
        "raw_evidence_exactly_once": all(prompt.prompt.count(message) == 1 for prompt in prompts for message in CASE_BY_ID[prompt.case_id].raw_delivered_messages),
        "model_visible_source_message_field_count": sum(prompt.prompt.count("source_message=") for prompt in prompts),
        "evidence_ref_preserved": all('evidence_ref="raw_message_1"' in prompt.second_order_block for prompt in prompts if prompt.condition == NEUTRAL_SECOND),
        "valid_actions_frozen": all(dict(prompt.sections)["VALID ACTIONS"] == dict(parents[prompt.prompt_id].sections)["VALID ACTIONS"] for prompt in prompts),
        "token_parity_by_neutral_prompt": parity,
        "all_token_parity_passed": all(parity.values()),
        "real_execution_authorized": False,
        "rank_one_is_final_scientific_selection": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _initialize(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in {
        "placebo_candidate_summary.json": {"status": "runtime_failure", "calibration_version": manifest["calibration_version"], "real_execution_authorized": False},
        "dp5_candidate_panel.json": [],
        "dp7_first_candidate_panel.json": [],
        "dp7_second_candidate_panel.json": [],
        "rejected_legacy_candidates.json": _legacy_rejections(),
        "search_diagnostics.json": {"status": "not_started"},
        "dependency_versions.json": {"status": "not_loaded"},
        "immutable_input_hashes.json": {"status": "not_verified"},
        "execution_environment.json": {"python": sys.version, "platform": platform.platform(), "github_run_id": os.getenv("GITHUB_RUN_ID"), "github_sha": os.getenv("GITHUB_SHA"), "model_or_api_execution": False},
    }.items():
        _write_json(output_dir / filename, value)
    (output_dir / "candidate_token_records.jsonl").write_text("")


def execute(output_dir: Path) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    _initialize(output_dir, manifest)
    try:
        immutable = verify_immutable_inputs()
        _write_json(output_dir / "immutable_input_hashes.json", immutable)
        raw_encoding, harmony_encode, versions = load_tokenizers({"dependencies": {"tiktoken": "0.13.0", "openai-harmony": "0.0.8"}})
        versions["pytest"] = importlib.metadata.version("pytest")
        if versions != manifest["dependencies"]:
            raise RuntimeError("dependency versions differ from manifest")
        _write_json(output_dir / "dependency_versions.json", versions)
        panels = {}
        diagnostics = {}
        for search_name, payload_length in (("DP5_first_order", 6), ("DP7_first_order", 4), ("DP7_second_order", 6)):
            panel, diagnostic = _search(search_name, payload_length, SEARCH_BOUNDS[search_name], raw_encoding.encode, harmony_encode)
            panels[search_name] = panel
            diagnostics[search_name] = diagnostic
        _write_json(output_dir / "dp5_candidate_panel.json", panels["DP5_first_order"])
        _write_json(output_dir / "dp7_first_candidate_panel.json", panels["DP7_first_order"])
        _write_json(output_dir / "dp7_second_candidate_panel.json", panels["DP7_second_order"])
        _write_json(output_dir / "search_diagnostics.json", diagnostics)
        with (output_dir / "candidate_token_records.jsonl").open("w") as stream:
            for search_name in ("DP5_first_order", "DP7_first_order", "DP7_second_order"):
                for record in panels[search_name]:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        insufficient = [name for name, panel in panels.items() if len(panel) < PANEL_SIZE]
        if not insufficient:
            status = "success"
            audit = global_audit(construct_rank_one_prompts(panels["DP5_first_order"][0], panels["DP7_first_order"][0], panels["DP7_second_order"][0]), raw_encoding.encode, harmony_encode)
            if not (audit["prompt_count"] == 16 and audit["exactly_six_neutral_prompts_changed"] and audit["reactive_prompts_byte_identical"] and audit["informative_prompts_byte_identical"] and audit["hidden_truth_leak_count"] == 0 and audit["forbidden_label_leak_count"] == 0 and audit["duplicated_raw_evidence_count"] == 0 and audit["raw_evidence_exactly_once"] and audit["model_visible_source_message_field_count"] == 0 and audit["evidence_ref_preserved"] and audit["valid_actions_frozen"] and audit["all_token_parity_passed"]):
                raise RuntimeError("rank-one global audit failed")
        else:
            audit = None
            status = "multiple_insufficient_panels" if len(insufficient) > 1 else {
                "DP5_first_order": "insufficient_dp5_candidates",
                "DP7_first_order": "insufficient_dp7_first_candidates",
                "DP7_second_order": "insufficient_dp7_second_candidates",
            }[insufficient[0]]
        _write_json(output_dir / "placebo_candidate_summary.json", {
            "status": status,
            "calibration_version": manifest["calibration_version"],
            "panel_sizes": {name: len(panel) for name, panel in panels.items()},
            "rank_one_global_audit": audit,
            "final_scientific_candidate_selected": False,
            "scientific_hypothesis_inference": None,
            "real_execution_authorized": False,
        })
        return 0 if status == "success" else 2
    except Exception as error:
        _write_json(output_dir / "placebo_candidate_summary.json", {"status": "runtime_failure", "calibration_version": manifest["calibration_version"], "error_type": type(error).__name__, "error_message": str(error), "scientific_hypothesis_inference": None, "real_execution_authorized": False})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return execute(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
