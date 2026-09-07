"""Structured, tokenizer-only feasibility search for first-order fillers."""

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
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .decision_point_prompting import (
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    FILLER_FORBIDDEN,
    HIDDEN_FIELD_NAMES,
    NEUTRAL_FIRST,
    NEUTRAL_SECOND,
    REACTIVE,
    forbidden_label_count,
)
from .decision_point_prompting_v0_2_1 import render_all_prompts as render_parent_prompts
from .decision_point_scenarios import CASE_BY_ID
from .run_decision_point_tokenizer_calibration_v0_2_2 import (
    IMMUTABLE_INPUTS as V021_IMMUTABLE_INPUTS,
    _block_shape,
    _record,
    _replace_section,
    load_tokenizers,
    pair_stream,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name(
    "decision_point_first_order_feasibility_manifest_v0_2_3.json"
)
CALIBRATION_VERSION = "btom-v2-decision-point-first-order-feasibility-0.2.3"
STRUCTURED_ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ0123456789_"
STRUCTURED_PATTERN = re.compile(r"^[BCDFGHJKLMNPQRSTVWXYZ0123456789_]+$")
MAX_DP5_CANDIDATES = 250000
MAX_DP7_CANDIDATES = 250000
SECOND_ORDER_VARIANT = 3087
EXPECTED_SECOND_ORDER_FILLERS = {8: "989_____", 10: "989_____ZZ"}
ADDITIONAL_IMMUTABLE_INPUTS = {
    "btom_v2/decision_point_tokenizer_calibration_manifest_v0_2_2.json":
        "18940981899f4338e23fdab74da830484294afe9d66339b70dc8a98e907718c2",
    "btom_v2/run_decision_point_tokenizer_calibration_v0_2_2.py":
        "4b6733199b2b802f9c7c5265ae6fb776fad2da33301d8383c19a10c2a9cb36ab",
}
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
    expected = {**V021_IMMUTABLE_INPUTS, **ADDITIONAL_IMMUTABLE_INPUTS}
    records = {}
    failures = []
    for relative, digest in expected.items():
        actual = _sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        matches = actual == digest
        records[relative] = {
            "expected_sha256": digest,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            failures.append(relative)
    parent = json.loads(
        (ROOT / "btom_v2/decision_point_prompt_protocol_v0_2_1.json").read_text()
    )
    records["parent_protocol_authorization_valid"] = (
        parent.get("real_execution_authorized") is False
    )
    if failures or not records["parent_protocol_authorization_valid"]:
        raise RuntimeError("immutable input verification failed: " + ", ".join(failures))
    return records


def _semantically_opaque(candidate: str) -> bool:
    if not candidate or not STRUCTURED_PATTERN.fullmatch(candidate):
        return False
    if candidate in {"A", "B", "C"}:
        return False
    forbidden = set(FILLER_FORBIDDEN) | {
        "DP5", "DP7", "CASE", "CONDITION", "PRODUCTIVE", "NECESSARY",
        "UNNECESSARY", "APPROPRIATE", "TARGET", "AGENT", "ACTION",
        "PROPOSITION", "SCORE", "STAGING", "CORRIDOR", "VICTIM",
    }
    upper = candidate.upper()
    return not any(term in upper for term in forbidden)


def periodic_candidates(length: int) -> Iterator[tuple[str, str]]:
    """Yield structured periodic strings using only the frozen alphabet."""
    for seed_length in range(1, 5):
        for symbols in itertools.product(STRUCTURED_ALPHABET, repeat=seed_length):
            seed = "".join(symbols)
            candidate = (seed * ((length + seed_length - 1) // seed_length))[:length]
            if _semantically_opaque(candidate):
                yield "periodic_structured", candidate


def vocabulary_fragments(raw_encoding: Any) -> tuple[str, ...]:
    """Collect deterministic, single-token ASCII fragments from the tokenizer."""
    retained = []
    for token_id in range(raw_encoding.n_vocab):
        try:
            text = raw_encoding.decode_single_token_bytes(token_id).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        if (
            text.isascii()
            and STRUCTURED_PATTERN.fullmatch(text)
            and _semantically_opaque(text)
            and list(raw_encoding.encode(text)) == [token_id]
        ):
            retained.append(text)
    return tuple(dict.fromkeys(retained))


def vocabulary_compositions(
    fragments: tuple[str, ...], length: int
) -> Iterator[tuple[str, str]]:
    """Compose exact-length candidates from tokenizer vocabulary fragments."""
    usable = tuple(fragment for fragment in fragments if len(fragment) <= length)
    for fragment in usable:
        if len(fragment) == length:
            yield "tokenizer_vocabulary_fragments", fragment
    for left in usable:
        remaining = length - len(left)
        for right in usable:
            if len(right) == remaining:
                candidate = left + right
                if _semantically_opaque(candidate):
                    yield "tokenizer_vocabulary_fragments", candidate
    for fragment in usable:
        seed = fragment + "_"
        candidate = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
        if _semantically_opaque(candidate):
            yield "tokenizer_hybrid", candidate


def structured_candidates(
    raw_encoding: Any, length: int, maximum: int
) -> Iterator[tuple[str, str]]:
    """Interleave both candidate families and deduplicate before evaluation."""
    fragments = vocabulary_fragments(raw_encoding)
    families: tuple[Iterable[tuple[str, str]], ...] = (
        periodic_candidates(length),
        vocabulary_compositions(fragments, length),
    )
    seen = set()
    active = [iter(family) for family in families]
    while active and len(seen) < maximum:
        next_active = []
        for iterator in active:
            try:
                family, candidate = next(iterator)
            except StopIteration:
                continue
            next_active.append(iterator)
            if candidate not in seen:
                seen.add(candidate)
                yield family, candidate
                if len(seen) >= maximum:
                    return
        active = next_active


@lru_cache(maxsize=1)
def _parents():
    return {prompt.prompt_id: prompt for prompt in render_parent_prompts()}


def _neutral_first(parent, filler: str):
    case = CASE_BY_ID[parent.case_id]
    key, value = case.first_order_representation[0]
    if len(filler) != len(value):
        raise ValueError("first-order filler length mismatch")
    return _replace_section(
        parent, "FIRST-ORDER REPRESENTATION", key, filler, f"first_order.{key}"
    )


def _neutral_second(parent, filler: str):
    value = CASE_BY_ID[parent.case_id].second_order_representation[0].believed_value
    if len(filler) != len(value):
        raise ValueError("second-order filler length mismatch")
    return _replace_section(
        parent,
        "SECOND-ORDER REPRESENTATION",
        "believed_value",
        filler,
        "second_order.0.believed_value",
    )


def _token_counts(prompt, raw_encode, harmony_encode) -> tuple[int, int]:
    return len(list(raw_encode(prompt.prompt))), len(list(harmony_encode(prompt.prompt)))


def _evaluate_dp5(
    long_filler: str, raw_encode: Callable, harmony_encode: Callable
) -> dict[str, Any]:
    parents = _parents()
    pair = ("DP5a_self_belief_false", "DP5b_self_belief_current")
    fillers = {pair[0]: long_filler, pair[1]: long_filler[:8]}
    details = {}
    requirements = {"prefix": long_filler.startswith(long_filler[:8])}
    for label in pair:
        neutral = _neutral_first(parents[f"{label}:{NEUTRAL_FIRST}"], fillers[label])
        explicit = parents[f"{label}:{EXPLICIT_FIRST}"]
        neutral_counts = _token_counts(neutral, raw_encode, harmony_encode)
        explicit_counts = _token_counts(explicit, raw_encode, harmony_encode)
        details[label] = {"neutral": neutral_counts, "explicit": explicit_counts}
        requirements[f"{label}_structure"] = (
            _block_shape(neutral.first_order_block) == _block_shape(explicit.first_order_block)
        )
        requirements[f"{label}_raw"] = neutral_counts[0] == explicit_counts[0]
        requirements[f"{label}_harmony"] = neutral_counts[1] == explicit_counts[1]
    requirements["raw_pair_difference"] = (
        details[pair[0]]["neutral"][0] - details[pair[1]]["neutral"][0]
        == details[pair[0]]["explicit"][0] - details[pair[1]]["explicit"][0]
    )
    requirements["harmony_pair_difference"] = (
        details[pair[0]]["neutral"][1] - details[pair[1]]["neutral"][1]
        == details[pair[0]]["explicit"][1] - details[pair[1]]["explicit"][1]
    )
    return {"requirements": requirements, "token_counts": details}


def _evaluate_dp7_first(
    filler: str, raw_encode: Callable, harmony_encode: Callable
) -> dict[str, Any]:
    parents = _parents()
    pair = ("DP7a_partner_belief_stale", "DP7b_partner_belief_current")
    details = {}
    requirements = {"same_filler_for_pair": True}
    for label in pair:
        neutral = _neutral_first(parents[f"{label}:{NEUTRAL_FIRST}"], filler)
        explicit = parents[f"{label}:{EXPLICIT_FIRST}"]
        neutral_counts = _token_counts(neutral, raw_encode, harmony_encode)
        explicit_counts = _token_counts(explicit, raw_encode, harmony_encode)
        details[label] = {"neutral": neutral_counts, "explicit": explicit_counts}
        requirements[f"{label}_structure"] = (
            _block_shape(neutral.first_order_block) == _block_shape(explicit.first_order_block)
        )
        requirements[f"{label}_raw"] = neutral_counts[0] == explicit_counts[0]
        requirements[f"{label}_harmony"] = neutral_counts[1] == explicit_counts[1]
    requirements["raw_pair_difference"] = (
        details[pair[0]]["neutral"][0] - details[pair[1]]["neutral"][0]
        == details[pair[0]]["explicit"][0] - details[pair[1]]["explicit"][0]
    )
    requirements["harmony_pair_difference"] = (
        details[pair[0]]["neutral"][1] - details[pair[1]]["neutral"][1]
        == details[pair[0]]["explicit"][1] - details[pair[1]]["explicit"][1]
    )
    return {"requirements": requirements, "token_counts": details}


def _run_search(
    name: str,
    raw_encoding: Any,
    harmony_encode: Callable,
    length: int,
    maximum: int,
    evaluator: Callable[[str, Callable, Callable], dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    generated = evaluated = 0
    seen = set()
    failures: Counter[str] = Counter()
    closest = []
    selected = None
    fragments = vocabulary_fragments(raw_encoding)
    active = [
        iter(periodic_candidates(length)),
        iter(vocabulary_compositions(fragments, length)),
    ]
    while active and len(seen) < maximum:
        next_active = []
        round_candidates = []
        for iterator in active:
            try:
                round_candidates.append(next(iterator))
                next_active.append(iterator)
            except StopIteration:
                continue
        active = next_active
        for family, candidate in round_candidates:
            generated += 1
            if candidate in seen:
                continue
            seen.add(candidate)
            if len(seen) > maximum:
                break
            result = evaluator(candidate, raw_encoding.encode, harmony_encode)
            evaluated += 1
            failed = [key for key, passed in result["requirements"].items() if not passed]
            failures.update(failed)
            record = {
                "candidate_family": family,
                "candidate_text": candidate,
                "passed_requirement_count": len(result["requirements"]) - len(failed),
                "requirement_count": len(result["requirements"]),
                "failed_requirements": failed,
                "token_counts": result["token_counts"],
            }
            closest.append(record)
            closest = sorted(
                closest,
                key=lambda item: (-item["passed_requirement_count"], item["candidate_text"]),
            )[:50]
            if not failed:
                selected = {**record, "requirements": result["requirements"]}
                break
        if selected:
            break
    diagnostics = {
        "search": name,
        "maximum_unique_candidates": maximum,
        "candidates_generated": generated,
        "candidates_deduplicated": len(seen),
        "candidates_evaluated": evaluated,
        "failure_counts_by_requirement": dict(sorted(failures.items())),
        "top_50_closest_candidates": closest,
        "selected_filler": selected["candidate_text"] if selected else None,
    }
    return selected, diagnostics


def reproduce_second_order(raw_encode: Callable, harmony_encode: Callable) -> dict[str, Any]:
    recomputed = {
        length: pair_stream("DP7_SECOND_ORDER_PAIR", SECOND_ORDER_VARIANT, length)
        for length in EXPECTED_SECOND_ORDER_FILLERS
    }
    if recomputed != EXPECTED_SECOND_ORDER_FILLERS:
        raise RuntimeError(f"second-order filler reproduction failed: {recomputed}")
    parents = _parents()
    pair = ("DP7a_partner_belief_stale", "DP7b_partner_belief_current")
    details = {}
    for label in pair:
        value = CASE_BY_ID[label].second_order_representation[0].believed_value
        neutral = _neutral_second(
            parents[f"{label}:{NEUTRAL_SECOND}"], recomputed[len(value)]
        )
        explicit = parents[f"{label}:{EXPLICIT_SECOND}"]
        case = CASE_BY_ID[label]
        requirements = {
            "raw_tokens_match": _token_counts(neutral, raw_encode, harmony_encode)[0]
            == _token_counts(explicit, raw_encode, harmony_encode)[0],
            "harmony_tokens_match": _token_counts(neutral, raw_encode, harmony_encode)[1]
            == _token_counts(explicit, raw_encode, harmony_encode)[1],
            "raw_message_once": neutral.prompt.count(case.raw_delivered_messages[0]) == 1,
            "explicit_raw_message_once": explicit.prompt.count(case.raw_delivered_messages[0]) == 1,
            "source_message_absent": "source_message=" not in neutral.prompt,
            "explicit_source_message_absent": "source_message=" not in explicit.prompt,
            "evidence_ref_preserved": 'evidence_ref="raw_message_1"' in neutral.second_order_block,
        }
        details[label] = {
            "neutral_counts": _token_counts(neutral, raw_encode, harmony_encode),
            "explicit_counts": _token_counts(explicit, raw_encode, harmony_encode),
            "requirements": requirements,
        }
        if not all(requirements.values()):
            raise RuntimeError(f"second-order reproduction checks failed for {label}")
    first, second = pair
    pairwise = {
        "raw_difference_matches": (
            details[first]["neutral_counts"][0] - details[second]["neutral_counts"][0]
            == details[first]["explicit_counts"][0] - details[second]["explicit_counts"][0]
        ),
        "harmony_difference_matches": (
            details[first]["neutral_counts"][1] - details[second]["neutral_counts"][1]
            == details[first]["explicit_counts"][1] - details[second]["explicit_counts"][1]
        ),
    }
    if not all(pairwise.values()):
        raise RuntimeError("second-order pairwise reproduction failed")
    return {
        "namespace": "DP7_SECOND_ORDER_PAIR",
        "variant": SECOND_ORDER_VARIANT,
        "fillers": {str(key): value for key, value in recomputed.items()},
        "case_checks": details,
        "pairwise_checks": pairwise,
    }


def construct_final_prompts(dp5_long: str, dp7_first: str):
    result = []
    for parent in render_parent_prompts():
        candidate = parent
        if parent.condition == NEUTRAL_FIRST:
            filler = dp5_long if parent.family == "DP5" else dp7_first
            if parent.family == "DP5" and len(
                CASE_BY_ID[parent.case_id].first_order_representation[0][1]
            ) == 8:
                filler = dp5_long[:8]
            candidate = _neutral_first(parent, filler)
        elif parent.condition == NEUTRAL_SECOND:
            length = len(CASE_BY_ID[parent.case_id].second_order_representation[0].believed_value)
            candidate = _neutral_second(parent, EXPECTED_SECOND_ORDER_FILLERS[length])
        result.append(candidate)
    return tuple(result)


def global_audit(prompts) -> dict[str, Any]:
    parents = _parents()
    changed = {p.prompt_id for p in prompts if p.prompt != parents[p.prompt_id].prompt}
    return {
        "prompt_count": len(prompts),
        "changed_prompt_ids": sorted(changed),
        "changed_prompt_set_valid": changed == EXPECTED_CHANGED,
        "reactive_immutable": all(
            p.prompt == parents[p.prompt_id].prompt for p in prompts if p.condition == REACTIVE
        ),
        "informative_immutable": all(
            p.prompt == parents[p.prompt_id].prompt
            for p in prompts if p.condition in {EXPLICIT_FIRST, EXPLICIT_SECOND}
        ),
        "hidden_truth_leak_count": sum(
            sum(field in p.prompt for field in HIDDEN_FIELD_NAMES) for p in prompts
        ),
        "forbidden_label_leak_count": sum(forbidden_label_count(p.prompt) for p in prompts),
        "duplicated_raw_evidence_count": sum(
            max(0, p.prompt.count(message) - 1)
            for p in prompts
            for message in CASE_BY_ID[p.case_id].raw_delivered_messages
        ),
        "raw_evidence_exactly_once": all(
            p.prompt.count(message) == 1
            for p in prompts
            for message in CASE_BY_ID[p.case_id].raw_delivered_messages
        ),
        "model_visible_source_message_field_count": sum(
            p.prompt.count("source_message=") for p in prompts
        ),
        "actions_frozen": all(
            dict(p.sections)["VALID ACTIONS"]
            == dict(parents[p.prompt_id].sections)["VALID ACTIONS"]
            for p in prompts
        ),
        "real_execution_authorized": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _initialize(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    placeholders = {
        "feasibility_summary.json": {
            "status": "runtime_failure",
            "calibration_version": manifest["calibration_version"],
            "real_execution_authorized": False,
        },
        "dp5_selected_fillers.json": {"selected": None},
        "dp7_first_selected_filler.json": {"selected": None},
        "verified_second_order_fillers.json": {"verified": False},
        "dp5_search_diagnostics.json": {"status": "not_started"},
        "dp7_first_search_diagnostics.json": {"status": "not_started"},
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
    for filename, payload in placeholders.items():
        _write_json(output_dir / filename, payload)
    (output_dir / "prompt_token_records.jsonl").write_text("")


def execute(output_dir: Path) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    _initialize(output_dir, manifest)
    try:
        immutable = verify_immutable_inputs()
        _write_json(output_dir / "immutable_input_hashes.json", immutable)
        raw_encoding, harmony_encode, versions = load_tokenizers({
            "dependencies": {
                "tiktoken": manifest["dependencies"]["tiktoken"],
                "openai-harmony": manifest["dependencies"]["openai-harmony"],
            }
        })
        versions["pytest"] = importlib.metadata.version("pytest")
        if versions != manifest["dependencies"]:
            raise RuntimeError("resolved dependency versions differ from manifest")
        _write_json(output_dir / "dependency_versions.json", versions)
        second = reproduce_second_order(raw_encoding.encode, harmony_encode)
        _write_json(output_dir / "verified_second_order_fillers.json", second)

        dp5, dp5_diagnostics = _run_search(
            "DP5_first_order", raw_encoding, harmony_encode, 10,
            MAX_DP5_CANDIDATES, _evaluate_dp5,
        )
        _write_json(output_dir / "dp5_search_diagnostics.json", dp5_diagnostics)
        dp7, dp7_diagnostics = _run_search(
            "DP7_first_order", raw_encoding, harmony_encode, 8,
            MAX_DP7_CANDIDATES, _evaluate_dp7_first,
        )
        _write_json(output_dir / "dp7_first_search_diagnostics.json", dp7_diagnostics)
        if dp5:
            _write_json(output_dir / "dp5_selected_fillers.json", {
                "candidate_family": dp5["candidate_family"],
                "long_filler": dp5["candidate_text"],
                "short_filler": dp5["candidate_text"][:8],
            })
        if dp7:
            _write_json(output_dir / "dp7_first_selected_filler.json", {
                "candidate_family": dp7["candidate_family"],
                "filler": dp7["candidate_text"],
            })
        status = (
            "success" if dp5 and dp7 else
            "multiple_no_candidate" if not dp5 and not dp7 else
            "dp5_no_candidate" if not dp5 else
            "dp7_first_no_candidate"
        )
        audit = None
        if dp5 and dp7:
            prompts = construct_final_prompts(dp5["candidate_text"], dp7["candidate_text"])
            audit = global_audit(prompts)
            if not (
                audit["prompt_count"] == 16
                and audit["changed_prompt_set_valid"]
                and audit["reactive_immutable"]
                and audit["informative_immutable"]
                and audit["hidden_truth_leak_count"] == 0
                and audit["forbidden_label_leak_count"] == 0
                and audit["duplicated_raw_evidence_count"] == 0
                and audit["raw_evidence_exactly_once"]
                and audit["model_visible_source_message_field_count"] == 0
                and audit["actions_frozen"]
            ):
                raise RuntimeError("global prompt audit failed")
            with (output_dir / "prompt_token_records.jsonl").open("w") as stream:
                for prompt in prompts:
                    record = _record(
                        prompt, SECOND_ORDER_VARIANT, raw_encoding.encode, harmony_encode
                    )
                    stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        _write_json(output_dir / "feasibility_summary.json", {
            "status": status,
            "calibration_version": manifest["calibration_version"],
            "model": manifest["model"],
            "tokenizer": manifest["raw_tokenizer"],
            "dependency_versions": versions,
            "dp5_selected": dp5 is not None,
            "dp7_first_selected": dp7 is not None,
            "second_order_reproduced": True,
            "global_audit": audit,
            "scientific_hypothesis_inference": None,
            "real_execution_authorized": False,
        })
        return 0 if status == "success" else 2
    except Exception as error:
        _write_json(output_dir / "feasibility_summary.json", {
            "status": "runtime_failure",
            "calibration_version": manifest["calibration_version"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "scientific_hypothesis_inference": None,
            "real_execution_authorized": False,
        })
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return execute(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
