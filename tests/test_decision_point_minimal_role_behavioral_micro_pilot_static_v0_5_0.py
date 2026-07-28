import hashlib,json
from pathlib import Path
from btom_v2 import run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_0 as runner
ROOT=Path(__file__).parents[1]
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_minimal_role_behavioral_micro_pilot_manifest_v0_5_0.json').read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_minimal_role_behavioral_micro_pilot_v0_5_0.yml').read_text()
def test_manifest_hashes_and_configuration():
 assert {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in MANIFEST['immutable_input_sha256']}==MANIFEST['immutable_input_sha256']
 assert MANIFEST['api_configuration']=={'model':'openai/gpt-oss-20b','transport':'groq_rest_fallback','temperature':0,'top_p':1,'max_tokens':256,'system_messages':0,'developer_messages':0,'user_messages':1,'tools':None,'seed_parameter_absent':True,'seed_controlled':False,'retries':0}
 assert MANIFEST['execution_budget']['maximum_api_attempts']==16
 assert MANIFEST['execution_budget']['attempts_per_prompt']==1
 assert MANIFEST['success_rate_threshold'] is None and not MANIFEST['fallback_counted_as_behavior']
def test_direct_sources_and_pairs():
 source=(ROOT/'btom_v2/run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_0.py').read_text()
 assert 'from .decision_point_minimal_role_control_prompting_v0_4_2 import render_all_prompts, run_audit' in source
 assert 'from .decision_point_scenarios import Action, CASES, CASE_BY_ID, evaluate_action' in source
 assert 'from .action_parser import parse_action' in source
 assert runner.MAX_ATTEMPTS==16 and runner.RETRIES==0 and runner.DELAY_SECONDS==1
 assert len(runner.PRIMARY_PAIRS)==4
 assert 'client.client=None; client.transport=TRANSPORT' in source
 assert "'fallback_counted_as_behavior':False" in source
 assert 'significance' not in source and 'effect_size' not in source
def test_outputs_and_workflow():
 assert len(runner.OUTPUTS)==10
 assert "github.event.head_commit.message == 'Run frozen minimal-role behavioral micro-pilot [run-behavioral-v0.5.0]'" in WORKFLOW
 assert WORKFLOW.count('python -m btom_v2.run_decision_point_minimal_role_behavioral_micro_pilot_v0_5_0 --output-dir')==1
 assert 'if: always()' in WORKFLOW and 'pytest==8.4.2' in WORKFLOW
 assert 'pip install groq' not in WORKFLOW.lower()
 assert 'echo $GROQ_API_KEY' not in WORKFLOW and 'printenv' not in WORKFLOW
