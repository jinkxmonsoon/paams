import hashlib
import json
from pathlib import Path

from btom_v2 import run_decision_point_minimal_role_tokenizer_v0_4_3 as runner

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / 'btom_v2/decision_point_minimal_role_tokenizer_manifest_v0_4_3.json').read_text())
WORKFLOW = (ROOT / '.github/workflows/decision_point_minimal_role_tokenizer_v0_4_3.yml').read_text()


def synthetic(value=0):
    primary = {}
    for h, contrast in runner.CONTRASTS.items():
        primary[h] = {case: {'raw_token_count_delta': value, 'harmony_token_count_delta': value} for case in contrast['cases']}
    interactions = {h: {'measurements': {m: {'difference_of_pairwise_differences': value} for m in ('raw_tokens','harmony_tokens')}} for h in ('H1','H2')}
    return primary, interactions


def test_manifest_and_hashes():
    assert MANIFEST['calibration_version'] == runner.CALIBRATION_VERSION
    assert MANIFEST['status'] == 'tokenizer_only_no_model_execution'
    assert MANIFEST['technical_hypothesis'] == 'T-CAL-3'
    assert MANIFEST['dependencies'] == {'tiktoken':'0.13.0','openai-harmony':'0.0.8','pytest':'8.4.2'}
    assert MANIFEST['tokenizer_layout'] == {'raw_tokenizer':'o200k_harmony','harmony_encoding':'HARMONY_GPT_OSS','system_messages':0,'developer_messages':0,'user_messages':1,'assistant_generation_prefix':True}
    assert MANIFEST['prompt_population'] == {'DP5':6,'DP7':10,'total':16}
    assert MANIFEST['immutable_parent_sha256'] == runner.IMMUTABLE_PARENT_SHA256
    assert {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in runner.IMMUTABLE_PARENT_SHA256} == runner.IMMUTABLE_PARENT_SHA256


def test_contrasts_sign_interactions_and_classification():
    assert runner.CONTRASTS['H1']['treatment'] == 'explicit_self_belief'
    assert runner.CONTRASTS['H1']['reference'] == 'matched_decision_record'
    assert runner.CONTRASTS['H2']['treatment'] == 'self_belief_plus_partner_belief'
    assert runner.CONTRASTS['H2']['reference'] == 'self_belief_plus_message_record'
    zero = synthetic(0)
    assert runner.acceptance_vector(*zero) == [0] * 12
    assert runner.classify(*zero) == 'exact_primary_token_parity'
    nonzero = synthetic(1)
    assert runner.classify(*nonzero) == 'nonzero_primary_token_difference'
    assert runner.exit_code_for_classification('exact_primary_token_parity') == 0
    assert runner.exit_code_for_classification('nonzero_primary_token_difference') == 0
    assert runner.exit_code_for_classification('runtime_failure') == 1


def test_static_runner_contract():
    source = (ROOT / 'btom_v2/run_decision_point_minimal_role_tokenizer_v0_4_3.py').read_text()
    assert 'from .decision_point_minimal_role_control_prompting_v0_4_2 import' in source
    assert 'render_all_prompts, run_audit' in source
    assert source.index('verify_immutable_inputs()') < source.index('load_tokenizers(manifest)')
    assert source.count('import tiktoken') == 1
    assert 'tiktoken.get_encoding("o200k_harmony")' in source
    assert 'Conversation.from_messages([Message.from_role_and_content(Role.USER, text)])' in source
    assert runner.OUTPUT_FILENAMES == {'minimal_role_tokenizer_summary.json','primary_contrast_token_deltas.json','interaction_token_diagnostics.json','v0_4_2_audit_snapshot.json','dependency_versions.json','immutable_input_hashes.json','execution_environment.json','prompt_token_records.jsonl','rendered_prompts.jsonl','block_token_records.jsonl'}
    assert 'requests' not in source and 'httpx' not in source
    assert MANIFEST['tolerance_threshold'] is None
    assert not any(MANIFEST[k] for k in ('wording_search_authorized','prompt_modification_authorized','candidate_generation_authorized','filler_authorized','padding_authorized','model_or_api_execution','real_execution_authorized'))


def test_workflow_activation():
    assert "github.event.head_commit.message == 'Measure minimal-role tokenizer parity [run-minimal-role-tokenizer-v0.4.3]'" in WORKFLOW
    assert WORKFLOW.count('python -m btom_v2.run_decision_point_minimal_role_tokenizer_v0_4_3 --output-dir') == 1
    assert 'PYTHONDONTWRITEBYTECODE=1 python -m pytest -q' in WORKFLOW
    assert 'if: always()' in WORKFLOW
    assert 'tiktoken==0.13.0 openai-harmony==0.0.8' in WORKFLOW
    assert 'pytest==8.4.2' in WORKFLOW
    assert '${{ github.sha }}' in WORKFLOW
    assert 'secret' not in WORKFLOW.lower()
