"""Tokenizer-only calibration for pair-coupled decision-point prompt fillers.

This module never contacts a model or API. Third-party tokenizer imports are
deliberately deferred until after immutable inputs have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .decision_point_prompting import (
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    FILLER_FORBIDDEN,
    HIDDEN_FIELD_NAMES,
    NEUTRAL_FIRST,
    NEUTRAL_SECOND,
    REACTIVE,
    RenderedPrompt,
    forbidden_label_count,
)
from .decision_point_prompting_v0_2_1 import render_all_prompts as render_parent_prompts
from .decision_point_scenarios import CASE_BY_ID


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_tokenizer_calibration_manifest_v0_2_2.json"
)
PARENT_PROTOCOL_PATH = Path(__file__).with_name(
    "decision_point_prompt_protocol_v0_2_1.json"
)
IMMUTABLE_INPUTS = {
    "btom_v2/decision_point_prompting_v0_2_1.py":
        "a2a6140e4e7676187c589a7e10c61fc6ff1b96f5aa0ba90258d54bb265e8b80f",
    "btom_v2/decision_point_prompt_protocol_v0_2_1.json":
        "9acec8dba50f0e33082cb8b0b238b8f4d27875b6ee4382fdee7c7e4ea806e8f7",
    "btom_v2/audit_decision_point_prompts_v0_2_1.py":
        "d4f7f9dbf0eda3219716987b379776207f1ad0e8c8131e704ec88a99ddcf55ec",
    "btom_v2/decision_point_prompt_amendment_v0_2_1.md":
        "032acd30bf00e562530c849b47a0c76608bbd07a1fdfe3982a51c092832054d9",
}
CALIBRATION_VERSION = "btom-v2-decision-point-tokenizer-calibration-0.2.2"
PAIR_NAMESPACES = (
    "DP5_FIRST_ORDER_PAIR",
    "DP7_FIRST_ORDER_PAIR",
    "DP7_SECOND_ORDER_PAIR",
)
FILLER_ALPHABET = "QZX789_"
EXPECTED_CHANGED = {
    "DP5a_self_belief_false:reactive_neutral_first_order_matched",
    "DP5b_self_belief_current:reactive_neutral_first_order_matched",
    "DP7a_partner_belief_stale:reactive_neutral_first_order_matched",
    "DP7b_partner_belief_current:reactive_neutral_first_order_matched",
    "DP7a_partner_belief_stale:first_order_neutral_second_order_matched",
    "DP7b_partner_belief_current:first_order_neutral_second_order_matched",
}


@dataclass(frozen=True)
class TokenRecord:
    prompt_id: str
    case_family: str
    condition: str
    prompt_sha256: str
    character_count: int
    raw_prompt_token_count: int
    harmony_input_token_count: int
    raw_token_id_sha256: str
    harmony_token_id_sha256: str
    first_order_filler: str | None
    second_order_filler: str | None
    filler_variant_index: int | None
    filler_pair_namespace: tuple[str, ...]
    pair_prefix_property: bool | None
    raw_message_occurrence_count: int
    model_visible_source_message_count: int
    hidden_field_leak_count: int
    forbidden_label_leak_count: int
    valid_action_count: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_immutable_inputs() -> dict[str, Any]:
    records = {}
    failures = []
    for relative_path, expected in IMMUTABLE_INPUTS.items():
        path = ROOT / relative_path
        actual = _sha256(path) if path.is_file() else None
        matches = actual == expected
        records[relative_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            failures.append(relative_path)
    protocol = json.loads(PARENT_PROTOCOL_PATH.read_text())
    authorization_valid = protocol.get("real_execution_authorized") is False
    records["parent_protocol_real_execution_authorized"] = False
    records["parent_protocol_authorization_valid"] = authorization_valid
    if failures or not authorization_valid:
        raise RuntimeError(
            "immutable input verification failed: "
            + ", ".join(failures or ["parent protocol authorization"])
        )
    return records


def load_tokenizers(manifest: dict[str, Any]):
    # Runtime-only imports: static validation and py_compile require no packages.
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
    }
    if versions != manifest["dependencies"]:
        raise RuntimeError(
            f"dependency version mismatch: expected {manifest['dependencies']}, got {versions}"
        )
    raw_encoding = tiktoken.get_encoding("o200k_harmony")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def harmony_tokens(prompt: str) -> list[int]:
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.USER, prompt)
        ])
        return list(
            harmony_encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
        )

    return raw_encoding, harmony_tokens, versions


def pair_stream(namespace: str, variant: int, length: int) -> str:
    if namespace not in PAIR_NAMESPACES:
        raise ValueError("unregistered pair namespace")
    if not 0 <= variant <= 9999 or length < 0:
        raise ValueError("variant or length out of range")
    output: list[str] = []
    block = 0
    while len(output) < length:
        digest = hashlib.sha256(f"{namespace}:{variant}:{block}".encode()).digest()
        output.extend(FILLER_ALPHABET[value % len(FILLER_ALPHABET)] for value in digest)
        block += 1
    filler = "".join(output[:length])
    if any(term in filler for term in FILLER_FORBIDDEN) or filler in {"A", "B", "C"}:
        raise ValueError("generated filler is not semantically opaque")
    return filler


def _replace_section(
    parent: RenderedPrompt,
    section_name: str,
    field_name: str,
    filler: str,
    filler_key: str,
) -> RenderedPrompt:
    old_block = dict(parent.sections)[section_name]
    pattern = rf'(?m)^({re.escape(field_name)}=")[^"]*(")$'
    new_block, substitutions = re.subn(pattern, rf"\g<1>{filler}\g<2>", old_block)
    if substitutions != 1 or len(new_block) != len(old_block):
        raise RuntimeError(f"failed exact-length replacement in {parent.prompt_id}")
    sections = tuple(
        (name, new_block if name == section_name else text)
        for name, text in parent.sections
    )
    return replace(
        parent,
        prompt="\n\n".join(text for _, text in sections),
        sections=sections,
        first_order_block=(
            new_block if section_name == "FIRST-ORDER REPRESENTATION"
            else parent.first_order_block
        ),
        second_order_block=(
            new_block if section_name == "SECOND-ORDER REPRESENTATION"
            else parent.second_order_block
        ),
        fillers=((filler_key, filler),),
    )


def construct_candidate(variant: int) -> tuple[RenderedPrompt, ...]:
    result = []
    for parent in render_parent_prompts():
        case = CASE_BY_ID[parent.case_id]
        candidate = parent
        if parent.condition == NEUTRAL_FIRST:
            namespace = (
                "DP5_FIRST_ORDER_PAIR" if parent.family == "DP5"
                else "DP7_FIRST_ORDER_PAIR"
            )
            key, value = case.first_order_representation[0]
            filler = pair_stream(namespace, variant, len(value))
            candidate = _replace_section(
                parent,
                "FIRST-ORDER REPRESENTATION",
                key,
                filler,
                f"first_order.{key}",
            )
        elif parent.condition == NEUTRAL_SECOND:
            value = case.second_order_representation[0].believed_value
            filler = pair_stream("DP7_SECOND_ORDER_PAIR", variant, len(value))
            candidate = _replace_section(
                parent,
                "SECOND-ORDER REPRESENTATION",
                "believed_value",
                filler,
                "second_order.0.believed_value",
            )
        result.append(candidate)
    return tuple(result)


def _token_hash(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _block_shape(block: str) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    lines = block.splitlines()
    keys = tuple(line.split("=", 1)[0] for line in lines[1:])
    punctuation = tuple(re.sub(r'"[^"]*"', '""', line) for line in lines[1:])
    return len(block), keys, punctuation


def _record(
    prompt: RenderedPrompt,
    variant: int,
    raw_encode: Callable[[str], list[int]],
    harmony_encode: Callable[[str], list[int]],
) -> TokenRecord:
    case = CASE_BY_ID[prompt.case_id]
    raw_ids = list(raw_encode(prompt.prompt))
    harmony_ids = list(harmony_encode(prompt.prompt))
    fillers = dict(prompt.fillers)
    first = next((value for key, value in fillers.items() if key.startswith("first_order.")), None)
    second = next((value for key, value in fillers.items() if key.startswith("second_order.")), None)
    namespaces = []
    if first:
        namespaces.append(
            "DP5_FIRST_ORDER_PAIR" if prompt.family == "DP5" else "DP7_FIRST_ORDER_PAIR"
        )
    if second:
        namespaces.append("DP7_SECOND_ORDER_PAIR")
    raw_occurrences = sum(
        prompt.prompt.count(message) for message in case.raw_delivered_messages
    )
    return TokenRecord(
        prompt_id=prompt.prompt_id,
        case_family=prompt.family,
        condition=prompt.condition,
        prompt_sha256=hashlib.sha256(prompt.prompt.encode()).hexdigest(),
        character_count=len(prompt.prompt),
        raw_prompt_token_count=len(raw_ids),
        harmony_input_token_count=len(harmony_ids),
        raw_token_id_sha256=_token_hash(raw_ids),
        harmony_token_id_sha256=_token_hash(harmony_ids),
        first_order_filler=first,
        second_order_filler=second,
        filler_variant_index=variant if fillers else None,
        filler_pair_namespace=tuple(namespaces),
        pair_prefix_property=True if fillers else None,
        raw_message_occurrence_count=raw_occurrences,
        model_visible_source_message_count=prompt.prompt.count("source_message="),
        hidden_field_leak_count=sum(field in prompt.prompt for field in HIDDEN_FIELD_NAMES),
        forbidden_label_leak_count=forbidden_label_count(prompt.prompt),
        valid_action_count=len(case.valid_actions),
    )


def evaluate_candidate(
    variant: int,
    raw_encode: Callable[[str], list[int]],
    harmony_encode: Callable[[str], list[int]],
) -> tuple[dict[str, Any], tuple[RenderedPrompt, ...], tuple[TokenRecord, ...]]:
    prompts = construct_candidate(variant)
    parents = {item.prompt_id: item for item in render_parent_prompts()}
    records = tuple(_record(item, variant, raw_encode, harmony_encode) for item in prompts)
    by_id = {record.prompt_id: record for record in records}
    by_prompt = {prompt.prompt_id: prompt for prompt in prompts}
    changed = {item.prompt_id for item in prompts if item.prompt != parents[item.prompt_id].prompt}

    def counts(case_id: str, condition: str) -> tuple[int, int]:
        record = by_id[f"{case_id}:{condition}"]
        return record.raw_prompt_token_count, record.harmony_input_token_count

    dp5_cases = ("DP5a_self_belief_false", "DP5b_self_belief_current")
    dp7_cases = ("DP7a_partner_belief_stale", "DP7b_partner_belief_current")
    dp5_fillers = [by_id[f"{case}:{NEUTRAL_FIRST}"].first_order_filler for case in dp5_cases]
    dp7_first_fillers = [by_id[f"{case}:{NEUTRAL_FIRST}"].first_order_filler for case in dp7_cases]
    dp7_second_fillers = [by_id[f"{case}:{NEUTRAL_SECOND}"].second_order_filler for case in dp7_cases]

    requirements = {
        "prompt_count": len(prompts) == 16,
        "exact_changed_prompt_set": changed == EXPECTED_CHANGED,
        "core_sections_frozen": all(
            all(
                dict(item.sections)[section] == dict(parents[item.prompt_id].sections)[section]
                for section in (
                    "TASK", "LOCAL OBSERVATION", "RAW DELIVERED MESSAGES",
                    "VALID ACTIONS", "OUTPUT FORMAT",
                )
            )
            for item in prompts
        ),
        "valid_actions_frozen": all(
            dict(item.sections)["VALID ACTIONS"]
            == dict(parents[item.prompt_id].sections)["VALID ACTIONS"]
            for item in prompts
        ),
        "reactive_immutability": all(
            item.prompt == parents[item.prompt_id].prompt
            for item in prompts if item.condition == REACTIVE
        ),
        "informative_immutability": all(
            item.prompt == parents[item.prompt_id].prompt
            for item in prompts if item.condition in {EXPLICIT_FIRST, EXPLICIT_SECOND}
        ),
        "dp5_first_block_structure": all(
            _block_shape(by_prompt[f"{case}:{NEUTRAL_FIRST}"].first_order_block)
            == _block_shape(by_prompt[f"{case}:{EXPLICIT_FIRST}"].first_order_block)
            for case in dp5_cases
        ),
        "dp5_raw_within_case": all(
            counts(case, NEUTRAL_FIRST)[0] == counts(case, EXPLICIT_FIRST)[0]
            for case in dp5_cases
        ),
        "dp5_harmony_within_case": all(
            counts(case, NEUTRAL_FIRST)[1] == counts(case, EXPLICIT_FIRST)[1]
            for case in dp5_cases
        ),
        "dp5_prefix": max(dp5_fillers, key=len).startswith(min(dp5_fillers, key=len)),
        "dp5_raw_pair_difference": (
            counts(dp5_cases[0], NEUTRAL_FIRST)[0] - counts(dp5_cases[1], NEUTRAL_FIRST)[0]
            == counts(dp5_cases[0], EXPLICIT_FIRST)[0] - counts(dp5_cases[1], EXPLICIT_FIRST)[0]
        ),
        "dp5_harmony_pair_difference": (
            counts(dp5_cases[0], NEUTRAL_FIRST)[1] - counts(dp5_cases[1], NEUTRAL_FIRST)[1]
            == counts(dp5_cases[0], EXPLICIT_FIRST)[1] - counts(dp5_cases[1], EXPLICIT_FIRST)[1]
        ),
        "dp7_first_identical": len(set(dp7_first_fillers)) == 1,
        "dp7_first_block_structure": all(
            _block_shape(by_prompt[f"{case}:{NEUTRAL_FIRST}"].first_order_block)
            == _block_shape(by_prompt[f"{case}:{EXPLICIT_FIRST}"].first_order_block)
            for case in dp7_cases
        ),
        "dp7_first_raw_within_case": all(
            counts(case, NEUTRAL_FIRST)[0] == counts(case, EXPLICIT_FIRST)[0]
            for case in dp7_cases
        ),
        "dp7_first_harmony_within_case": all(
            counts(case, NEUTRAL_FIRST)[1] == counts(case, EXPLICIT_FIRST)[1]
            for case in dp7_cases
        ),
        "dp7_second_block_structure": all(
            _block_shape(by_prompt[f"{case}:{NEUTRAL_SECOND}"].second_order_block)
            == _block_shape(by_prompt[f"{case}:{EXPLICIT_SECOND}"].second_order_block)
            and by_prompt[f"{case}:{NEUTRAL_SECOND}"].first_order_block
            == by_prompt[f"{case}:{EXPLICIT_SECOND}"].first_order_block
            and 'evidence_ref="raw_message_1"'
            in by_prompt[f"{case}:{NEUTRAL_SECOND}"].second_order_block
            for case in dp7_cases
        ),
        "dp7_second_raw_within_case": all(
            counts(case, NEUTRAL_SECOND)[0] == counts(case, EXPLICIT_SECOND)[0]
            for case in dp7_cases
        ),
        "dp7_second_harmony_within_case": all(
            counts(case, NEUTRAL_SECOND)[1] == counts(case, EXPLICIT_SECOND)[1]
            for case in dp7_cases
        ),
        "dp7_second_prefix": max(dp7_second_fillers, key=len).startswith(min(dp7_second_fillers, key=len)),
        "dp7_second_raw_pair_difference": (
            counts(dp7_cases[0], NEUTRAL_SECOND)[0] - counts(dp7_cases[1], NEUTRAL_SECOND)[0]
            == counts(dp7_cases[0], EXPLICIT_SECOND)[0] - counts(dp7_cases[1], EXPLICIT_SECOND)[0]
        ),
        "dp7_second_harmony_pair_difference": (
            counts(dp7_cases[0], NEUTRAL_SECOND)[1] - counts(dp7_cases[1], NEUTRAL_SECOND)[1]
            == counts(dp7_cases[0], EXPLICIT_SECOND)[1] - counts(dp7_cases[1], EXPLICIT_SECOND)[1]
        ),
        "no_hidden_leaks": sum(record.hidden_field_leak_count for record in records) == 0,
        "no_forbidden_labels": sum(record.forbidden_label_leak_count for record in records) == 0,
        "raw_evidence_not_duplicated": all(
            record.raw_message_occurrence_count == 1
            for record in records if record.case_family == "DP7"
        ),
        "source_message_absent": sum(
            record.model_visible_source_message_count for record in records
        ) == 0,
    }
    return {
        "variant": variant,
        "passed_requirement_count": sum(requirements.values()),
        "requirement_count": len(requirements),
        "failed_requirements": [name for name, passed in requirements.items() if not passed],
        "requirements": requirements,
        "changed_prompt_ids": sorted(changed),
        "token_counts": {
            record.prompt_id: {
                "raw": record.raw_prompt_token_count,
                "harmony": record.harmony_input_token_count,
            }
            for record in records
        },
    }, prompts, records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _initialize_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "selected_fillers.json", {"selected_variant": None, "fillers": {}})
    (output_dir / "prompt_token_records.jsonl").write_text("")
    _write_json(output_dir / "search_diagnostics.json", {"candidates_tested": 0, "status": "not_started"})
    _write_json(output_dir / "dependency_versions.json", {"status": "not_loaded"})
    _write_json(output_dir / "immutable_input_hashes.json", {"status": "not_verified"})
    _write_json(output_dir / "execution_environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_sha": os.getenv("GITHUB_SHA"),
        "model_or_api_execution": False,
    })
    _write_json(output_dir / "calibration_summary.json", {
        "status": "runtime_failure",
        "calibration_version": manifest["calibration_version"],
        "real_execution_authorized": False,
    })


def execute(output_dir: Path) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    _initialize_artifacts(output_dir, manifest)
    try:
        immutable = verify_immutable_inputs()
        _write_json(output_dir / "immutable_input_hashes.json", immutable)
        raw_encoding, harmony_encode, versions = load_tokenizers(manifest)
        _write_json(output_dir / "dependency_versions.json", versions)

        failure_counts: Counter[str] = Counter()
        best: list[dict[str, Any]] = []
        selected = None
        for variant in range(
            manifest["search_range"]["minimum_variant"],
            manifest["search_range"]["maximum_variant"] + 1,
        ):
            result, prompts, records = evaluate_candidate(
                variant, raw_encoding.encode, harmony_encode
            )
            failure_counts.update(result["failed_requirements"])
            best.append(result)
            best = sorted(
                best,
                key=lambda item: (-item["passed_requirement_count"], item["variant"]),
            )[:20]
            if not result["failed_requirements"]:
                selected = (result, prompts, records)
                break

        candidates_tested = (selected[0]["variant"] + 1) if selected else 10000
        diagnostics = {
            "status": "success" if selected else "no_candidate",
            "candidates_tested": candidates_tested,
            "constraint_failure_counts": dict(sorted(failure_counts.items())),
            "top_20_candidates": best,
        }
        _write_json(output_dir / "search_diagnostics.json", diagnostics)
        if not selected:
            _write_json(output_dir / "calibration_summary.json", {
                "status": "no_candidate",
                "calibration_version": manifest["calibration_version"],
                "model": manifest["model"],
                "tokenizer": manifest["raw_tokenizer"],
                "resolved_package_versions": versions,
                "prompt_count": 16,
                "candidates_tested": candidates_tested,
                "selected_variant": None,
                "real_execution_authorized": False,
            })
            return 2

        result, prompts, records = selected
        selected_fillers = {
            namespace: {
                str(length): pair_stream(namespace, result["variant"], length)
                for length in sorted({
                    len(value)
                    for prompt in prompts
                    for key, value in prompt.fillers
                    if (
                        (namespace == "DP5_FIRST_ORDER_PAIR" and prompt.family == "DP5" and key.startswith("first_order."))
                        or (namespace == "DP7_FIRST_ORDER_PAIR" and prompt.family == "DP7" and key.startswith("first_order."))
                        or (namespace == "DP7_SECOND_ORDER_PAIR" and key.startswith("second_order."))
                    )
                })
            }
            for namespace in PAIR_NAMESPACES
        }
        _write_json(output_dir / "selected_fillers.json", {
            "selected_variant": result["variant"],
            "fillers": selected_fillers,
        })
        with (output_dir / "prompt_token_records.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        requirements = result["requirements"]
        summary = {
            "status": "success",
            "calibration_version": manifest["calibration_version"],
            "model": manifest["model"],
            "tokenizer": manifest["raw_tokenizer"],
            "resolved_package_versions": versions,
            "prompt_count": len(prompts),
            "candidates_tested": candidates_tested,
            "selected_variant": result["variant"],
            "pair_coupled_filler_checks": {
                key: value for key, value in requirements.items()
                if "prefix" in key or "identical" in key
            },
            "raw_token_checks": {
                key: value for key, value in requirements.items() if "raw_" in key
            },
            "harmony_token_checks": {
                key: value for key, value in requirements.items() if "harmony_" in key
            },
            "changed_prompt_ids": result["changed_prompt_ids"],
            "hidden_truth_leak_count": sum(r.hidden_field_leak_count for r in records),
            "forbidden_label_leak_count": sum(r.forbidden_label_leak_count for r in records),
            "duplicated_raw_evidence_count": sum(
                max(0, r.raw_message_occurrence_count - 1)
                for r in records if r.case_family == "DP7"
            ),
            "model_visible_source_message_field_count": sum(
                r.model_visible_source_message_count for r in records
            ),
            "real_execution_authorized": False,
            "scientific_hypothesis_inference": None,
        }
        _write_json(output_dir / "calibration_summary.json", summary)
        return 0
    except Exception as error:
        _write_json(output_dir / "calibration_summary.json", {
            "status": "runtime_failure",
            "calibration_version": manifest["calibration_version"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "real_execution_authorized": False,
        })
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
