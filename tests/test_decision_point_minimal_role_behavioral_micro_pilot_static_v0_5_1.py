import hashlib,json
from pathlib import Path
from btom_v2 import run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1 as runner
from btom_v2.decision_point_minimal_role_control_prompting_v0_4_2 import render_all_prompts
from btom_v2.decision_point_scenarios import CASE_BY_ID
ROOT=Path(__file__).parents[1]
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_minimal_role_behavioral_micro_pilot_manifest_v0_5_1.json').read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_minimal_role_behavioral_micro_pilot_v0_5_1.yml').read_text()
def test_frozen_inputs_population_and_pairs():
 assert {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in MANIFEST['immutable_input_sha256']}==MANIFEST['immutable_input_sha256']
 prompts=render_all_prompts(); assert len(prompts)==16
 assert runner.MAX_REQUESTS==16 and runner.RETRIES==0 and len(runner.PRIMARY_PAIRS)==4
 assert MANIFEST['parent_provenance']['result']=='interface_failure_completion_budget_exhaustion'
def test_exact_body_and_schema():
 for prompt in render_all_prompts():
  case=CASE_BY_ID[prompt.case_id]; body=runner.request_body(prompt.prompt,case)
  assert set(body)=={'model','messages','temperature','top_p','max_completion_tokens','reasoning_effort','reasoning_format','response_format','stream'}
  assert body['model']=='openai/gpt-oss-20b' and body['messages']==[{'role':'user','content':prompt.prompt}]
  assert body['temperature']==0 and body['top_p']==1 and body['max_completion_tokens']==1024
  assert body['reasoning_effort']=='low' and body['reasoning_format']=='hidden' and body['stream'] is False
  assert 'max_tokens' not in body and 'tools' not in body and 'seed' not in body
  fmt=body['response_format']; schema=fmt['json_schema']['schema']
  assert fmt['type']=='json_schema' and fmt['json_schema']['strict'] is True
  assert schema['required']==['action','target','message','reason'] and schema['additionalProperties'] is False
  assert schema['properties']['action']['enum']==sorted({a.action for a in case.valid_actions})
  assert schema['properties']['target']['enum']==sorted({a.target for a in case.valid_actions})
 by={(p.case_id,p.condition):p for p in render_all_prompts()}
 for _,case,ref,tr in runner.PRIMARY_PAIRS:
  assert runner.schema_hash(CASE_BY_ID[case])==runner.schema_hash(CASE_BY_ID[case])
  assert by[(case,ref)].prompt != '' and by[(case,tr)].prompt != ''
def test_source_telemetry_and_no_fallback():
 source=(ROOT/'btom_v2/run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1.py').read_text()
 assert 'render_all_prompts, run_audit' in source and 'parse_action' in source and 'evaluate_action' in source
 for field in ('raw_api_response_json','message_content','empty_visible_content','message_reasoning','finish_reason','usage','reasoning_token_details','completion_output_token_count','latency_seconds','response_format_schema_hash'):
  assert field in source
 assert "'fallback_counted_as_behavior':False" in source
 assert "'reasoning_is_behavioral_evidence':False" in source
 assert 'significance' not in source and 'effect_size' not in source
def test_workflow_secret_scope_and_order():
 assert "github.event.head_commit.message == 'Repair frozen behavioral output interface [run-behavioral-v0.5.1]'" in WORKFLOW
 assert WORKFLOW.count('python -m btom_v2.run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1 --output-dir')==1
 assert WORKFLOW.index('python -m pytest -q') < WORKFLOW.index('python -m btom_v2.run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1 --output-dir')
 assert WORKFLOW.count('GROQ_API_KEY')==2 and '      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}' in WORKFLOW and 'if: always()' in WORKFLOW
 assert 'pip install groq' not in WORKFLOW.lower()
