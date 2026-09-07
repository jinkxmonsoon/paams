"""Tokenizer-only measurement of the frozen v0.4.2 minimal-role prompts."""

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
    EXPLICIT_SELF_BELIEF, FIRST_ORDER_TITLE, MATCHED_DECISION_RECORD,
    MESSAGE_RECORD, PARTNER_BELIEF, SECOND_ORDER_TITLE,
)
from .decision_point_minimal_role_control_prompting_v0_4_2 import (
    render_all_prompts, run_audit,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("decision_point_minimal_role_tokenizer_manifest_v0_4_3.json")
CALIBRATION_VERSION = "btom-v2-decision-point-minimal-role-tokenizer-0.4.3"
IMMUTABLE_PARENT_SHA256 = {
    "btom_v2/decision_point_minimal_role_control_manifest_v0_4_2.json": "73786ac184545ffaec749fcda195c68ceb17404ff890a5849b989f3b068c6318",
    "btom_v2/decision_point_minimal_role_control_prompting_v0_4_2.py": "0557df5c97efcf2bf4176f7c6ca36ea2202f18735bc1e6eed44807034894ec68",
    "btom_v2/decision_point_minimal_role_control_design_v0_4_2.md": "818c79a7f3148f542161e0e48cc8ee033f9560de147850f93039c57825910f4a",
    "tests/test_decision_point_minimal_role_control_v0_4_2.py": "03dc6920f8516af04a05a0e92a18a61dc78bee02f41caca52d2730648521e477",
    "btom_v2/decision_point_content_matched_control_manifest_v0_4_0.json": "5a45397e0d420e188b86b2520bc3a701e4dd3e84c6aa4d6d2508b821937999b6",
    "btom_v2/decision_point_content_matched_control_prompting_v0_4_0.py": "ab9c6b01308df15560db4be1a735bb5cef31a101942ff79431b38ce6b0733ccf",
    "btom_v2/decision_point_content_matched_control_design_v0_4_0.md": "581d1e769c792365e8c89dff48093de9af788fc145365d7e6b984e8f66d62c27",
    "btom_v2/decision_point_natural_control_prompting_v0_3_0.py": "08df09f8e5d950f0fde422993133ca59e40362bcd06ba799446951a5701c1961",
    "btom_v2/decision_point_scenarios.py": "e7a936cca701356dc2f9a5c8ab9621b5e593af932b854afe4c9a146acf1994f5",
}
AUDIT_REQUIREMENTS = {
    "prompt_count": 16, "DP5_prompt_count": 6, "DP7_prompt_count": 10,
    "only_representation_role_changed_from_v0_4_0": True,
    "common_sections_byte_identical_to_v0_4_0": True,
    "observations_actions_messages_output_frozen": True,
    "pairwise_audit_count": 6, "pairwise_audits_passed": True,
    "independent_H2_provenance_preserved": True, "DP7_raw_evidence_exactly_once": True,
    "hidden_state_leak_count": 0, "scoring_label_leak_count": 0,
    "condition_name_leak_count": 0, "source_message_field_count": 0,
    "legacy_marker_count": 0, "role_values_ascii_six_characters": True,
    "tokenizer_counts_produced": False, "tokenizer_parity_claimed": False,
    "model_or_api_execution": False, "real_execution_authorized": False,
}
CONTRASTS = {
    "H1": {"treatment": EXPLICIT_SELF_BELIEF, "reference": MATCHED_DECISION_RECORD,
           "cases": ("DP5a_self_belief_false", "DP5b_self_belief_current"), "block": FIRST_ORDER_TITLE},
    "H2": {"treatment": PARTNER_BELIEF, "reference": MESSAGE_RECORD,
           "cases": ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"), "block": SECOND_ORDER_TITLE},
}
OUTPUT_FILENAMES = {
    "minimal_role_tokenizer_summary.json", "primary_contrast_token_deltas.json",
    "interaction_token_diagnostics.json", "v0_4_2_audit_snapshot.json",
    "dependency_versions.json", "immutable_input_hashes.json", "execution_environment.json",
    "prompt_token_records.jsonl", "rendered_prompts.jsonl", "block_token_records.jsonl",
}

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_immutable_inputs() -> dict[str, Any]:
    records = {}
    for relative, expected in IMMUTABLE_PARENT_SHA256.items():
        actual = _sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        records[relative] = {"expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected}
    failures = [path for path, record in records.items() if not record["matches"]]
    if failures:
        raise RuntimeError("immutable input verification failed: " + ", ".join(failures))
    return records

def load_tokenizers(manifest: dict[str, Any]):
    import tiktoken
    from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, load_harmony_encoding
    versions = {name: importlib.metadata.version(name) for name in ("tiktoken", "openai-harmony", "pytest")}
    if versions != manifest["dependencies"]:
        raise RuntimeError(f"dependency version mismatch: expected {manifest['dependencies']}, got {versions}")
    raw = tiktoken.get_encoding("o200k_harmony")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    def harmony_tokens(text: str) -> list[int]:
        conversation = Conversation.from_messages([Message.from_role_and_content(Role.USER, text)])
        return list(harmony_encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))
    return raw, harmony_tokens, versions

def _token_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

def _measure(text: str, raw: Callable[[str], list[int]], harmony: Callable[[str], list[int]]) -> dict[str, Any]:
    raw_ids, harmony_ids = list(raw(text)), list(harmony(text))
    return {"character_count": len(text), "utf8_byte_count": len(text.encode()),
            "line_count": len(text.splitlines()), "approximate_whitespace_token_count": len(text.split()),
            "raw_token_count": len(raw_ids), "raw_token_id_sha256": _token_hash(raw_ids),
            "harmony_token_count": len(harmony_ids), "harmony_token_id_sha256": _token_hash(harmony_ids)}

def measure_prompts(prompts, raw, harmony):
    prompt_records, rendered, blocks = [], [], []
    for prompt in prompts:
        sections = dict(prompt.sections)
        names = tuple(name for name in (FIRST_ORDER_TITLE, SECOND_ORDER_TITLE) if name in sections)
        prompt_hash = hashlib.sha256(prompt.prompt.encode()).hexdigest()
        prompt_records.append({"prompt_id": prompt.prompt_id, "case_id": prompt.case_id, "family": prompt.family,
            "condition": prompt.condition, "complete_rendered_prompt": prompt.prompt, "prompt_sha256": prompt_hash,
            **_measure(prompt.prompt, raw, harmony), "included_representation_block_names": names,
            "structural_audit_status": "passed_v0_4_2_frozen_audit", "model_or_api_execution": False})
        rendered.append({"prompt_id": prompt.prompt_id, "prompt": prompt.prompt, "prompt_sha256": prompt_hash})
        for title in names:
            text = sections[title]
            fields = dict(line.split("=", 1) for line in text.splitlines()[1:])
            blocks.append({"prompt_id": prompt.prompt_id, "case_id": prompt.case_id, "condition": prompt.condition,
                "block_title": title, "block_text": text,
                "representation_role": fields["representation_role"].strip('"'),
                "represented_value": fields["represented_value"].strip('"'), **_measure(text, raw, harmony),
                "block_only_harmony_diagnostic_only": True,
                "block_only_harmony_warning": "Diagnostic only: the independent Harmony conversation envelope prevents additive reconstruction of complete-prompt counts."})
    return prompt_records, rendered, blocks

def primary_contrast_deltas(prompt_records, block_records):
    prompts = {(r["case_id"], r["condition"]): r for r in prompt_records}
    blocks = {(r["case_id"], r["condition"], r["block_title"]): r for r in block_records}
    fields = ("raw_token_count", "harmony_token_count", "character_count", "utf8_byte_count", "line_count")
    result = {}
    for hypothesis, contrast in CONTRASTS.items():
        result[hypothesis] = {}
        for case in contrast["cases"]:
            treatment, reference = prompts[(case, contrast["treatment"])], prompts[(case, contrast["reference"])]
            tb, rb = blocks[(case, contrast["treatment"], contrast["block"])], blocks[(case, contrast["reference"], contrast["block"])]
            result[hypothesis][case] = {"formula": f'{contrast["treatment"]} minus {contrast["reference"]}',
                **{f"{field}_delta": treatment[field] - reference[field] for field in fields},
                "block_only_raw_token_delta": tb["raw_token_count"] - rb["raw_token_count"],
                "block_only_harmony_token_delta": tb["harmony_token_count"] - rb["harmony_token_count"]}
    return result

def interaction_diagnostics(prompt_records):
    prompts = {(r["case_id"], r["condition"]): r for r in prompt_records}
    result = {}
    for hypothesis, contrast in CONTRASTS.items():
        first, second = contrast["cases"]
        measurements = {}
        for label, field in {"raw_tokens": "raw_token_count", "harmony_tokens": "harmony_token_count",
                             "characters": "character_count", "utf8_bytes": "utf8_byte_count"}.items():
            treatment = prompts[(first, contrast["treatment"])][field] - prompts[(second, contrast["treatment"])][field]
            reference = prompts[(first, contrast["reference"])][field] - prompts[(second, contrast["reference"])][field]
            measurements[label] = {"treatment_first_case_minus_second_case": treatment,
                "reference_first_case_minus_second_case": reference,
                "difference_of_pairwise_differences": treatment - reference}
        result[hypothesis] = {"formula": "(treatment first case - treatment second case) - (reference first case - reference second case)",
                              "components": {"first_case": first, "second_case": second,
                                             "treatment": contrast["treatment"], "reference": contrast["reference"]},
                              "measurements": measurements, "scientific_hypothesis_test": False}
    return result

def acceptance_vector(primary, interactions) -> list[int]:
    values = []
    for hypothesis in ("H1", "H2"):
        for case in CONTRASTS[hypothesis]["cases"]:
            values += [primary[hypothesis][case]["raw_token_count_delta"], primary[hypothesis][case]["harmony_token_count_delta"]]
    values += [interactions[h]["measurements"][m]["difference_of_pairwise_differences"]
               for h in ("H1", "H2") for m in ("raw_tokens", "harmony_tokens")]
    return values

def classify(primary, interactions) -> str:
    return "exact_primary_token_parity" if acceptance_vector(primary, interactions) == [0] * 12 else "nonzero_primary_token_difference"

def exit_code_for_classification(classification: str) -> int:
    return 1 if classification == "runtime_failure" else 0

def _json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def _jsonl(path: Path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))

def _initialize(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename in OUTPUT_FILENAMES:
        if filename.endswith(".jsonl"):
            (output_dir / filename).write_text("")
        else:
            _json(output_dir / filename, {"classification": "runtime_failure", "model_or_api_execution": False,
                                          "real_execution_authorized": False} if filename == "minimal_role_tokenizer_summary.json" else {})

def execute(output_dir: Path) -> int:
    _initialize(output_dir)
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
        immutable = verify_immutable_inputs()
        _json(output_dir / "immutable_input_hashes.json", immutable)
        audit = run_audit()
        _json(output_dir / "v0_4_2_audit_snapshot.json", audit)
        failed = [key for key, expected in AUDIT_REQUIREMENTS.items() if audit.get(key) != expected]
        if len(audit.get("pairwise_audits", ())) != 6 or not all(record.get("passed") for record in audit.get("pairwise_audits", ())):
            failed.append("pairwise_audits")
        if failed:
            raise RuntimeError("v0.4.2 structural audit failed: " + ", ".join(failed))
        prompts = render_all_prompts()
        if (len(prompts), sum(p.family == "DP5" for p in prompts), sum(p.family == "DP7" for p in prompts)) != (16, 6, 10):
            raise RuntimeError("frozen prompt population mismatch")
        raw, harmony, versions = load_tokenizers(manifest)
        _json(output_dir / "dependency_versions.json", versions)
        prompt_records, rendered, blocks = measure_prompts(prompts, raw.encode, harmony)
        _jsonl(output_dir / "prompt_token_records.jsonl", prompt_records)
        _jsonl(output_dir / "rendered_prompts.jsonl", rendered)
        _jsonl(output_dir / "block_token_records.jsonl", blocks)
        primary, interactions = primary_contrast_deltas(prompt_records, blocks), interaction_diagnostics(prompt_records)
        _json(output_dir / "primary_contrast_token_deltas.json", primary)
        _json(output_dir / "interaction_token_diagnostics.json", interactions)
        classification = classify(primary, interactions)
        vector = acceptance_vector(primary, interactions)
        _json(output_dir / "execution_environment.json", {"python": sys.version, "platform": platform.platform(),
              "github_run_id": os.getenv("GITHUB_RUN_ID"), "github_sha": os.getenv("GITHUB_SHA"), "model_or_api_execution": False})
        _json(output_dir / "minimal_role_tokenizer_summary.json", {"classification": classification,
              "exact_parity": classification == "exact_primary_token_parity", "acceptance_vector": vector,
              "primary_deltas": primary, "interaction_diagnostics": interactions,
              "prompt_counts": {"DP5": 6, "DP7": 10, "total": 16}, "dependency_versions": versions,
              "scientific_inference": None, "model_or_api_execution": False, "real_execution_authorized": False})
        return exit_code_for_classification(classification)
    except Exception as error:
        _json(output_dir / "minimal_role_tokenizer_summary.json", {"classification": "runtime_failure",
              "error_type": type(error).__name__, "error_message": str(error), "scientific_inference": None,
              "model_or_api_execution": False, "real_execution_authorized": False})
        return 1

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return execute(parser.parse_args().output_dir)

if __name__ == "__main__":
    raise SystemExit(main())
