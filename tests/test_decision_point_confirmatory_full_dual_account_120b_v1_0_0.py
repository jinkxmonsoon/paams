import hashlib,json
from pathlib import Path
import pytest
from btom_v2 import run_decision_point_confirmatory_full_dual_account_120b_v1_0_0 as replication
from btom_v2 import run_decision_point_confirmatory_full_dual_account_v1_0_0 as engine
from btom_v2.analyze_decision_point_confirmatory_v1_0_0 import analyze
from btom_v2.decision_point_confirmatory_prompting_v1_0_0 import canonical_prompt_digest,request_order
from btom_v2.run_decision_point_confirmatory_v1_0_0 import body
ROOT=Path(__file__).resolve().parents[1]
MANIFEST_20B=json.loads((ROOT/'btom_v2/decision_point_confirmatory_full_dual_account_manifest_v1_0_0.json').read_text())
MANIFEST_120B=json.loads(replication.MANIFEST.read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_confirmatory_full_dual_account_120b_v1_0_0.yml').read_text()
DUMMY={'GROQ_API_KEY':'dummy-primary-secret-120b','GROQ_API_KEY_SECONDARY':'dummy-secondary-secret-120b'}

def success_payload(p):
 a,t=p.valid_actions[0];return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'action':a,'target':t,'message':'','reason':'mock'})}}],'x_groq':{'seed':p.seed},'system_fingerprint':'mock','service_tier':'default','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
class Factory:
 def __init__(self,behavior=None):self.behavior=behavior or (lambda slot,p,n:(True,200,success_payload(p),None,.01));self.calls=[];self.slot_counts={'primary':0,'secondary':0};self.exhausted=set()
 def __call__(self,slot,key,all_credentials):
  factory=self
  class Client:
   model=replication.MODEL
   def call(self,p):
    assert slot not in factory.exhausted;assert replication.request_body(p)['model']==replication.MODEL
    factory.slot_counts[slot]+=1;factory.calls.append((slot,p.prompt_id,replication.request_body(p)));return factory.behavior(slot,p,len(factory.calls))
  return Client()
def run(tmp_path,monkeypatch,behavior=None,sha='0123456789abcdef0123456789abcdef01234567',name='run'):
 monkeypatch.setattr(engine,'load_tokenizers',lambda manifest:(list,list,manifest['dependencies']))
 env={**DUMMY};
 if sha is not None:env['GITHUB_SHA']=sha
 factory=Factory(behavior);out=tmp_path/name;code=replication.execute(out,sleep=lambda _:None,client_factory=factory,environ=env);return code,out,factory

def test_freeze_assignment_api_and_analyzer_identity():
 prompts_20b=engine.selected_prompts();prompts_120b=request_order()
 assert len(prompts_20b)==len(prompts_120b)==432 and len({p.variant_id for p in prompts_120b})==72
 for a,b in zip(prompts_20b,prompts_120b):assert (a.prompt_id,a.prompt,a.seed,a.valid_actions)==(b.prompt_id,b.prompt,b.seed,b.valid_actions)
 assert canonical_prompt_digest(prompts_120b)==replication.PROMPT_DIGEST
 assert hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest()==replication.SCENARIO_HASH
 assert MANIFEST_120B['frozen_selected_order']==MANIFEST_20B['frozen_selected_order']
 assert MANIFEST_120B['credential_routing']==MANIFEST_20B['credential_routing']
 assert engine.assignment_audit(prompts_120b,MANIFEST_120B['credential_routing']['variant_assignment'])==engine.assignment_audit(prompts_20b,MANIFEST_20B['credential_routing']['variant_assignment'])
 assert {k:(MANIFEST_20B['api'][k],MANIFEST_120B['api'][k]) for k in MANIFEST_20B['api'] if MANIFEST_20B['api'][k]!=MANIFEST_120B['api'][k]}=={'model':('openai/gpt-oss-20b','openai/gpt-oss-120b')}
 for p in prompts_120b:
  expected=body(p).copy();expected['model']=replication.MODEL;assert replication.request_body(p)==expected
 analyzer=(ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py').read_bytes()
 assert hashlib.sha1(f'blob {len(analyzer)}\0'.encode()+analyzer).hexdigest()==replication.ANALYZER_GIT_BLOB

def test_manifest_precommitment_and_no_outcome_adaptation():
 provenance=MANIFEST_120B['replication_provenance'];source=Path(replication.__file__).read_text()
 assert provenance['replication_precommitted'] and provenance['replication_mandatory_regardless_of_20b_result']
 assert provenance['same_bank_as_20b'] and provenance['same_analysis_as_20b'] and not provenance['selection_uses_20b_treatment_outcomes']
 for forbidden in ('behavioral_results.jsonl','raw_api_responses.jsonl','confirmatory_full_dual_analysis.json','role_sensitive_counts','H1_primary_interpretable'):assert forbidden not in source
 assert 'engine.execute(' in source and 'engine.MANIFEST=MANIFEST' in source and 'engine.COLLECTION_BATCH=COLLECTION_BATCH' in source

def test_complete_mock_analysis_provenance_and_redaction(tmp_path,monkeypatch):
 code,out,factory=run(tmp_path,monkeypatch);assert code==0 and len(factory.calls)==432 and all(c[2]['model']==replication.MODEL for c in factory.calls)
 summary=json.loads((out/'confirmatory_full_dual_summary.json').read_text());records=[json.loads(x) for x in (out/'confirmatory_full_dual_call_records.jsonl').read_text().splitlines()]
 assert summary['operational_classification']=='full_confirmatory_batch_complete' and summary['ready_for_final_analysis']
 assert summary['canonical_prompt_count']==summary['canonical_complete_calls']==432 and summary['complete_variants_by_family']=={'H1':36,'H2':36}
 assert sum(summary['variants_by_final_credential_slot'].values())==72 and summary['all_variants_credential_homogeneous'] and summary['failover_count']==0 and summary['scientific_inference'] is None
 actual=json.loads((out/'confirmatory_full_dual_analysis.json').read_text());actual.pop('credential_slot_diagnostic');assert actual==analyze(records)
 p=json.loads((out/'confirmatory_full_dual_replication_provenance.json').read_text());assert p['actual_execution_sha']=='0123456789abcdef0123456789abcdef01234567'
 assert p==replication.provenance({**DUMMY,'GITHUB_SHA':p['actual_execution_sha']})
 artifact=''.join(x.read_text(errors='replace') for x in out.iterdir() if x.is_file())
 for secret in DUMMY.values():assert secret not in artifact and secret[:12] not in artifact and hashlib.sha256(secret.encode()).hexdigest() not in artifact

def test_provenance_without_github_sha(tmp_path,monkeypatch):
 code,out,_=run(tmp_path,monkeypatch,sha=None,name='no-sha');assert code==0
 p=json.loads((out/'confirmatory_full_dual_replication_provenance.json').read_text());assert p['actual_execution_sha'] is None
 assert p['parent_20b_run_id']==34036092185 and p['parent_20b_execution_sha']=='e290e07b1a60ea12befb7f6c76a2fc7f0997a132' and p['parent_20b_artifact_id']==9992434005 and p['parent_20b_artifact_zip_sha256']=='0d87972d686e7cbc54aca181bd390add0af5fee335afe927473d0894d10e332e'

def tpd_first(first_slot):
 fired=False
 def behavior(slot,p,n):
  nonlocal fired
  if slot==first_slot and not fired:fired=True;return False,429,None,json.dumps({'error':{'message':'tokens per day limit reached'}}),.01
  return True,200,success_payload(p),None,.01
 return behavior
@pytest.mark.parametrize('first_slot',['primary','secondary'])
def test_single_tpd_replays_whole_variant_symmetrically(tmp_path,monkeypatch,first_slot):
 code,out,factory=run(tmp_path,monkeypatch,tpd_first(first_slot),name='tpd-'+first_slot);assert code==0
 attempts=[json.loads(x) for x in (out/'confirmatory_full_dual_transport_attempts.jsonl').read_text().splitlines()];trigger=next(i for i,a in enumerate(attempts) if a['confirmed_tpd_exhaustion'])
 assert all(a['attempted_credential_slot']!=first_slot for a in attempts[trigger+1:])
 summary=json.loads((out/'confirmatory_full_dual_summary.json').read_text());assert summary['canonical_prompt_count']==432 and summary['all_variants_credential_homogeneous'] and summary['ready_for_final_analysis']

def dual_tpd(first_slot):
 first_fired=False;second_fired=False
 def behavior(slot,p,n):
  nonlocal first_fired,second_fired
  if slot==first_slot and not first_fired:first_fired=True;return False,429,None,json.dumps({'error':{'message':'TPD quota exhausted'}}),.01
  if first_fired and slot!=first_slot and not second_fired:second_fired=True;return False,429,None,json.dumps({'error':{'message':'TPD quota exhausted'}}),.01
  return True,200,success_payload(p),None,.01
 return behavior
@pytest.mark.parametrize('first_slot',['primary','secondary'])
def test_dual_tpd_terminal_both_directions(tmp_path,monkeypatch,first_slot):
 code,out,_=run(tmp_path,monkeypatch,dual_tpd(first_slot),name='dual-'+first_slot);assert code==1
 summary=json.loads((out/'confirmatory_full_dual_summary.json').read_text());assert summary['dual_account_tpd_exhausted'] and not summary['ready_for_final_analysis'] and summary['scientific_inference'] is None
 assert not (out/'confirmatory_full_dual_analysis.json').exists()

@pytest.mark.parametrize(('label','failure'),[
 ('generic-429',(False,429,None,'generic rate limit',.01)),('http-500',(False,500,None,'server error',.01)),
 ('parse',(True,200,{'choices':[{'finish_reason':'stop','message':{'content':'not json'}}]},None,.01)),
 ('illegal',(True,200,{'choices':[{'finish_reason':'stop','message':{'content':'{"action":"illegal","target":"illegal","message":"","reason":"mock"}'}}]},None,.01))])
def test_non_tpd_failures_fail_closed_without_failover(tmp_path,monkeypatch,label,failure):
 def behavior(slot,p,n):return failure if n==1 else (True,200,success_payload(p),None,.01)
 code,out,_=run(tmp_path,monkeypatch,behavior,name=label);assert code==1
 attempts=[json.loads(x) for x in (out/'confirmatory_full_dual_transport_attempts.jsonl').read_text().splitlines()];assert len(attempts)==432 and not any(a['confirmed_tpd_exhaustion'] for a in attempts) and all(a['assigned_credential_slot']==a['attempted_credential_slot'] for a in attempts)
 summary=json.loads((out/'confirmatory_full_dual_summary.json').read_text());assert summary['operational_classification']=='full_confirmatory_batch_incomplete' and summary['failover_count']==0 and summary['exhausted_credential_slots']==[] and not summary['ready_for_final_analysis'];assert not (out/'confirmatory_full_dual_analysis.json').exists()

def test_wrapper_restores_engine_globals(tmp_path,monkeypatch):
 old=(engine.MANIFEST,engine.COLLECTION_BATCH);run(tmp_path,monkeypatch,name='restore');assert (engine.MANIFEST,engine.COLLECTION_BATCH)==old

def test_workflow_guard_scope_and_final_status():
 assert 'branches: [codex/execute-micro-pilot-on-current-branch-vsnbkd]' in WORKFLOW and 'timeout-minutes: 300' in WORKFLOW
 assert "github.event.head_commit.message == 'Execute mandatory 120B replication [experiment-v1.0.0]'" in WORKFLOW and 'Prepare mandatory 120B confirmatory replication [experiment-v1.0.0]' not in WORKFLOW
 execution=WORKFLOW[WORKFLOW.index('- name: Execute mandatory 120B replication once'):WORKFLOW.index('- uses: actions/upload-artifact@v4')]
 assert WORKFLOW.index('python -m pytest -q')<WORKFLOW.index('- name: Execute mandatory 120B replication once')
 assert 'GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}' in execution and 'GROQ_API_KEY_SECONDARY: ${{ secrets.GROQ_API_KEY_SECONDARY }}' in execution
 assert 'GROQ_API_KEY' not in WORKFLOW[:WORKFLOW.index('- name: Execute mandatory 120B replication once')]
 assert 'runner_exit_code=$?' in execution and '"$GITHUB_SHA" "$runner_exit_code"' in execution and 'exit "$runner_exit_code"' in execution
 assert 'if: always()' in WORKFLOW and 'retention-days: 30' in WORKFLOW and 'decision-point-confirmatory-full-dual-account-120b-v1.0.0-${{ github.run_id }}' in WORKFLOW
