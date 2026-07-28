import hashlib,json
from pathlib import Path
from btom_v2 import run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1 as runner
from btom_v2.decision_point_minimal_role_control_prompting_v0_4_2 import render_all_prompts
from btom_v2.decision_point_scenarios import CASE_BY_ID
ROOT=Path(__file__).parents[1]
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_minimal_role_behavioral_micro_pilot_manifest_v0_5_1.json').read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_minimal_role_behavioral_micro_pilot_v0_5_1.yml').read_text()
RUNNER=(ROOT/'btom_v2/run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1.py').read_text()
def test_frozen_inputs_population_and_pairs():
 assert {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in MANIFEST['immutable_input_sha256']}==MANIFEST['immutable_input_sha256']
 assert len(render_all_prompts())==16
 assert runner.MAX_REQUESTS==16 and runner.RETRIES==0 and len(runner.PRIMARY_PAIRS)==4
 assert MANIFEST['interface_amendment']['json_schema_changed'] is False
def test_exact_corrected_body_and_unchanged_schema():
 prompts=render_all_prompts(); by={(p.case_id,p.condition):p for p in prompts}
 for prompt in prompts:
  case=CASE_BY_ID[prompt.case_id]; body=runner.request_body(prompt.prompt,case)
  assert set(body)=={'model','messages','temperature','top_p','max_completion_tokens','reasoning_effort','include_reasoning','response_format','stream'}
  assert body['model']=='openai/gpt-oss-20b' and body['messages']==[{'role':'user','content':prompt.prompt}]
  assert body['temperature']==0 and body['top_p']==1 and body['max_completion_tokens']==1024
  assert body['reasoning_effort']=='low' and body['include_reasoning'] is False and body['stream'] is False
  assert not {'reasoning_format','max_tokens','tools','seed'}.intersection(body)
  fmt=body['response_format']; schema=fmt['json_schema']['schema']
  assert fmt['type']=='json_schema' and fmt['json_schema']['strict'] is True
  assert schema['required']==['action','target','message','reason'] and schema['additionalProperties'] is False
  assert schema['properties']['action']['enum']==sorted({a.action for a in case.valid_actions})
  assert schema['properties']['target']['enum']==sorted({a.target for a in case.valid_actions})
 for _,case,reference,treatment in runner.PRIMARY_PAIRS:
  reference_schema=runner.response_schema(CASE_BY_ID[by[(case,reference)].case_id])
  treatment_schema=runner.response_schema(CASE_BY_ID[by[(case,treatment)].case_id])
  assert json.dumps(reference_schema,sort_keys=True,separators=(',',':'))==json.dumps(treatment_schema,sort_keys=True,separators=(',',':'))
  assert runner.schema_hash(CASE_BY_ID[by[(case,reference)].case_id])==runner.schema_hash(CASE_BY_ID[by[(case,treatment)].case_id])
def test_reasoning_interface_finish_gate_and_telemetry():
 assert MANIFEST['api_configuration']['include_reasoning'] is False
 assert 'reasoning_format' not in RUNNER and 'reasoning_format' not in MANIFEST['api_configuration']
 assert "completed=finish=='stop'" in RUNNER
 assert "r['completed_finish_reason']" in RUNNER
 assert "exhausted=finish in {'length','max_tokens'}" in RUNNER
 for field in ('raw_api_response_json','message_content','empty_visible_content','message_reasoning','finish_reason','completed_finish_reason','usage','reasoning_token_details','completion_output_token_count','latency_seconds','response_format_schema_hash'):
  assert field in RUNNER
 assert "'fallback_counted_as_behavior':False" in RUNNER and "'reasoning_is_behavioral_evidence':False" in RUNNER
 assert 'significance' not in RUNNER and 'effect_size' not in RUNNER
def test_workflow_secret_scope_and_order():
 assert "github.event.head_commit.message == 'Correct GPT-OSS reasoning interface [run-behavioral-v0.5.1]'" in WORKFLOW
 command='python -m btom_v2.run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_1 --output-dir'
 assert WORKFLOW.count(command)==1
 assert WORKFLOW.index('python -m pytest -q') < WORKFLOW.index(command)
 assert WORKFLOW.count('GROQ_API_KEY')==2 and '      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}' in WORKFLOW
 assert 'if: always()' in WORKFLOW and 'pip install groq' not in WORKFLOW.lower()
