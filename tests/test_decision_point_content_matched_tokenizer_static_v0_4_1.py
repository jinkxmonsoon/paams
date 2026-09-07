import ast
import hashlib
import json
from pathlib import Path

from btom_v2.run_decision_point_content_matched_tokenizer_v0_4_1 import (
    CONTRASTS, IMMUTABLE_PARENT_SHA256, acceptance_vector, classify,
    exit_code_for_classification, interaction_diagnostics,
    primary_contrast_deltas, verify_immutable_inputs,
)
from btom_v2.decision_point_content_matched_control_prompting_v0_4_0 import render_all_prompts

ROOT=Path(__file__).parents[1]
RUNNER=ROOT/'btom_v2/run_decision_point_content_matched_tokenizer_v0_4_1.py'
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_content_matched_tokenizer_manifest_v0_4_1.json').read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_content_matched_tokenizer_v0_4_1.yml').read_text()
SOURCE=RUNNER.read_text()


def test_manifest_identity_dependencies_layout_and_flags():
    assert MANIFEST['calibration_version']=='btom-v2-decision-point-content-matched-tokenizer-0.4.1'
    assert MANIFEST['dependencies']=={'tiktoken':'0.13.0','openai-harmony':'0.0.8','pytest':'8.4.2'}
    assert MANIFEST['tokenizer_layout']=={'raw_tokenizer':'o200k_harmony','harmony_encoding':'HARMONY_GPT_OSS','system_messages':0,'developer_messages':0,'user_messages':1,'assistant_generation_prefix':True}
    assert MANIFEST['prompt_population']=={'DP5':6,'DP7':10,'total':16}
    assert MANIFEST['tolerance_threshold'] is None
    for key in ('wording_search_authorized','prompt_modification_authorized','candidate_generation_authorized','filler_authorized','padding_authorized','model_or_api_execution','real_execution_authorized'):
        assert MANIFEST[key] is False


def test_immutable_hashes_exact_everywhere():
    assert MANIFEST['immutable_parent_sha256']==IMMUTABLE_PARENT_SHA256
    assert verify_immutable_inputs()
    for path,digest in IMMUTABLE_PARENT_SHA256.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        assert f'{digest}  {path}' in WORKFLOW


def test_direct_frozen_renderer_and_exact_population():
    assert 'from .decision_point_content_matched_control_prompting_v0_4_0 import (' in SOURCE
    for name in ('render_all_prompts','run_audit','CONDITIONS_BY_FAMILY','FIRST_ORDER_TITLE','SECOND_ORDER_TITLE','MATCHED_DECISION_RECORD','EXPLICIT_SELF_BELIEF','MESSAGE_RECORD','PARTNER_BELIEF'):
        assert name in SOURCE
    prompts=render_all_prompts()
    assert len(prompts)==16
    assert sum(p.family=='DP5' for p in prompts)==6
    assert sum(p.family=='DP7' for p in prompts)==10
    assert '_section(' not in SOURCE and 'render_prompt(' not in SOURCE


def _records(delta=0):
    prompt=[]; blocks=[]
    for h,c in CONTRASTS.items():
        for case in c['cases']:
            for condition,block in ((c['reference'],c['reference_block']),(c['informative'],c['informative_block'])):
                add=delta if condition==c['informative'] else 0
                rec={'case_id':case,'condition':condition,'raw_token_count':100+add,'harmony_token_count':106+add,'character_count':500+add,'utf8_byte_count':500+add,'line_count':20}
                prompt.append(rec)
                blocks.append({**rec,'block_title':block,'raw_token_count':20+add,'harmony_token_count':26+add})
    return prompt,blocks


def test_exact_contrasts_interactions_and_twelve_value_vector():
    prompts,blocks=_records()
    primary=primary_contrast_deltas(prompts,blocks)
    interactions=interaction_diagnostics(prompts)
    assert primary['H1']['DP5a_self_belief_false']['formula']=='explicit_self_belief minus matched_decision_record'
    assert primary['H2']['DP7a_partner_belief_stale']['formula']=='self_belief_plus_partner_belief minus self_belief_plus_message_record'
    assert acceptance_vector(primary,interactions)==[0]*12
    assert classify(primary,interactions)=='exact_primary_token_parity'


def test_nonzero_completed_and_runtime_exit_semantics():
    prompts,blocks=_records(delta=1)
    primary=primary_contrast_deltas(prompts,blocks); interactions=interaction_diagnostics(prompts)
    vector=acceptance_vector(primary,interactions)
    assert len(vector)==12 and any(vector)
    assert classify(primary,interactions)=='nonzero_primary_token_difference'
    assert exit_code_for_classification('exact_primary_token_parity')==0
    assert exit_code_for_classification('nonzero_primary_token_difference')==0
    assert exit_code_for_classification('runtime_failure')==1


def test_interaction_formulas_are_difference_of_differences():
    prompts,blocks=_records()
    # perturb only DP5a treatment and DP7b treatment
    for r in prompts:
        if (r['case_id'],r['condition'])==('DP5a_self_belief_false','explicit_self_belief'): r['raw_token_count']+=2
        if (r['case_id'],r['condition'])==('DP7b_partner_belief_current','self_belief_plus_partner_belief'): r['harmony_token_count']+=3
    i=interaction_diagnostics(prompts)
    assert i['H1']['measurements']['raw_tokens']['difference_of_pairwise_differences']==2
    assert i['H2']['measurements']['harmony_tokens']['difference_of_pairwise_differences']==-3


def test_block_schema_and_diagnostic_warning_present():
    for field in ('prompt_id','case_id','condition','block_title','block_text','representation_role','represented_value','raw_token_count','raw_token_id_sha256','harmony_token_count','harmony_token_id_sha256'):
        assert f'"{field}"' in SOURCE
    warning=MANIFEST['block_only_harmony_warning']
    assert 'Diagnostic only' in warning and 'must not reconstruct complete-prompt counts' in warning
    assert warning in SOURCE


def test_runtime_only_tokenizers_no_clients_or_search():
    tree=ast.parse(SOURCE)
    top={a.name for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
    assert 'tiktoken' not in top and 'openai_harmony' not in top
    lower=SOURCE.lower()
    for bad in ('import groq','from groq','openai import','requests','httpx','urllib','api.groq.com','api.openai.com','candidate_specifications','padding','filler'):
        assert bad not in lower


def test_exact_workflow_guard_full_tests_single_measurement_always_upload():
    assert "github.event.head_commit.message ==\n          'Repair content-matched tokenizer activation [run-content-matched-tokenizer-v0.4.1]'" in WORKFLOW
    assert 'python -m pytest -q 2>&1 | tee content-matched-tokenizer-artifact/pytest.log' in WORKFLOW
    assert WORKFLOW.count('python -m btom_v2.run_decision_point_content_matched_tokenizer_v0_4_1')==1
    assert 'if: always()' in WORKFLOW and 'retention-days: 30' in WORKFLOW
    assert 'GROQ_API_KEY' not in WORKFLOW and 'OPENAI_API_KEY' not in WORKFLOW


def test_parent_workflow_assertion_is_version_scoped_and_amendment_is_infrastructure_only():
    parent_test = (ROOT / "tests/test_decision_point_content_matched_control_v0_4_0.py").read_text()
    assert 'glob("*content_matched*")' not in parent_test
    assert """assert not (
        ROOT
        / ".github/workflows"
        / "decision_point_content_matched_control_v0_4_0.yml"
    ).exists()""" in parent_test
    assert MANIFEST["immutable_parent_sha256"][
        "tests/test_decision_point_content_matched_control_v0_4_0.py"
    ] == "4cc27164df30ac9664807e564b65c57e32c6a3f01b24e64052a05c1c12172e5c"
    assert MANIFEST["parent_test_amendment"] == {
        "file": "tests/test_decision_point_content_matched_control_v0_4_0.py",
        "reason": "Scoped an obsolete repository-global no-workflow assertion to the v0.4.0 design workflow only.",
        "scientific_inputs_changed": False,
        "prompt_content_changed": False,
        "scenario_content_changed": False,
    }


def test_other_five_immutable_hashes_are_unchanged():
    assert {k: v for k, v in IMMUTABLE_PARENT_SHA256.items() if k != "tests/test_decision_point_content_matched_control_v0_4_0.py"} == {
        "btom_v2/decision_point_content_matched_control_manifest_v0_4_0.json": "5a45397e0d420e188b86b2520bc3a701e4dd3e84c6aa4d6d2508b821937999b6",
        "btom_v2/decision_point_content_matched_control_prompting_v0_4_0.py": "ab9c6b01308df15560db4be1a735bb5cef31a101942ff79431b38ce6b0733ccf",
        "btom_v2/decision_point_content_matched_control_design_v0_4_0.md": "581d1e769c792365e8c89dff48093de9af788fc145365d7e6b984e8f66d62c27",
        "btom_v2/decision_point_natural_control_prompting_v0_3_0.py": "08df09f8e5d950f0fde422993133ca59e40362bcd06ba799446951a5701c1961",
        "btom_v2/decision_point_scenarios.py": "e7a936cca701356dc2f9a5c8ab9621b5e593af932b854afe4c9a146acf1994f5",
    }
