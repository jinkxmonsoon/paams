"""Tests for the minimal v0.6.1 pre-execution amendment."""
import ast,json,hashlib
from pathlib import Path
import pytest
from btom_v2 import journal_j1_record_holdout_prompting_v0_6_0 as old
from btom_v2 import journal_j1_record_holdout_prompting_v0_6_1 as new
ROOT=Path(__file__).resolve().parents[1]; BTOM=ROOT/'btom_v2'

def test_parent_hashes_and_scenarios_unchanged():
 assert new.verify_parent_hashes()==new.PARENT_HASHES
 assert new.parent.scenarios()==json.loads((BTOM/'journal_j1_record_holdout_scenarios_v0_6_0.json').read_text())['scenarios']

def test_only_affordance_changes_and_both_entries_eligible():
 variants,prompts,manifest,audit=new.build(); old_prompts=[old.render(v,mr,mi) for v in variants for mr in (0,1) for mi in (0,1)]
 assert len(prompts)==72 and audit['passed']
 for before,after in zip(old_prompts,prompts,strict=True):
  assert before['prompt_text'].replace(new.OLD,new.NEW)==after['prompt_text']
  assert new.NEW in after['prompt_text'] and "about one entry in the current partner record" in after['prompt_text']
  assert after['prompt_text'].count(': ')>=2 and after['prompt_text'].index(old.FRAMING_LABEL)<after['prompt_text'].index('You have exactly one action')
 assert manifest['scenario_bank_changed'] is manifest['q_changed'] is manifest['r_changed'] is False

def test_seed_order_round_gate_and_stop_rule_identity():
 _,_,manifest,_=new.build(); parent=json.loads((BTOM/old.OUTPUTS[2]).read_text())
 assert manifest['frozen_seeds']==parent['frozen_seeds']
 for a,b in zip(manifest['frozen_request_order'],parent['frozen_request_order'],strict=True):
  assert {k:v for k,v in a.items() if k!='prompt_sha256'}=={k:v for k,v in b.items() if k!='prompt_sha256'}
 assert manifest['cell_sequences']==parent['cell_sequences'] and manifest['sequence_assignments']==parent['sequence_assignments']
 assert manifest['five_scenario_control_gates']==parent['five_scenario_control_gates']
 assert manifest['headroom_diagnostic']==parent['headroom_diagnostic'] and manifest['post_execution_classifications']==parent['post_execution_classifications']

def test_cell_invariance_record_only_and_no_leakage():
 variants,prompts,_,audit=new.build(); assert audit['checks']['cell_invariance']
 forbidden=('partner belief','m_r','m_i','q_true','r_true','relevant','irrelevant','gold action')
 assert all(p['F']=='R' and not any(x in p['prompt_text'].lower() for x in forbidden) for p in prompts)
 for v in variants: assert all(v['variant_id'] not in p['prompt_text'] for p in prompts)

def test_source_selection_behavior_network_and_secret_free():
 source=Path(new.__file__).read_text(); tree=ast.parse(source)
 assert 'journal_j1_confirmatory_selection_v0_3_0.json' not in source
 assert not any(x in source for x in ('behavioral_results','raw_api_response','GROQ_API_KEY','Authorization'))
 imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
 assert not imports & {'requests','httpx','openai','groq','socket','urllib'}

def test_deterministic_actions_artifacts(tmp_path):
 a=tmp_path/'a'; b=tmp_path/'b'; new.generate(a); new.generate(b)
 assert all((a/n).read_bytes()==(b/n).read_bytes() for n in new.OUTPUTS)

def test_harmony_validation_when_available(tmp_path):
 pytest.importorskip('openai_harmony'); new.generate(tmp_path)
 assert new.harmony_validate(tmp_path)
 data=json.loads((tmp_path/'journal_j1_record_holdout_harmony_prompt_audit_v0_6_1.json').read_text()); groups=json.loads((tmp_path/'journal_j1_record_holdout_harmony_variant_audit_v0_6_1.json').read_text())
 assert len(data)==72 and len(groups)==18 and all(r['max_abs_within_variant_difference']==0 for r in groups)

def test_harmony_fail_closed_on_one_variant(monkeypatch,tmp_path):
 class E:
  def render_conversation_for_completion(self,*a): return [1] * (10 + calls.pop(0))
 calls=[0,0,0,1]+[0]*68
 fake=type('M',(),{'Conversation':type('C',(),{'from_messages':staticmethod(lambda x:x)}),'HarmonyEncodingName':type('H',(),{'HARMONY_GPT_OSS':1}),'Message':type('M',(),{'from_role_and_content':staticmethod(lambda *x:x)}),'Role':type('R',(),{'USER':1,'ASSISTANT':2}),'load_harmony_encoding':lambda x:E()})
 import sys; monkeypatch.setitem(sys.modules,'openai_harmony',fake); new.generate(tmp_path); assert new.harmony_validate(tmp_path) is False
 assert json.loads((tmp_path/'workflow_status.json').read_text())['holdout_execution_ready'] is False
