import hashlib,json
from pathlib import Path
import pytest
from btom_v2 import run_decision_point_confirmatory_recovery_v1_0_0 as recovery
from btom_v2.decision_point_confirmatory_prompting_v1_0_0 import canonical_prompt_digest,request_order
from btom_v2.run_decision_point_confirmatory_v1_0_0 import body
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_confirmatory_recovery_manifest_v1_0_0.json').read_text());WORKFLOW=(ROOT/'.github/workflows/decision_point_confirmatory_recovery_v1_0_0.yml').read_text();DUMMY={'GROQ_API_KEY':'dummy-primary-secret','GROQ_API_KEY_SECONDARY':'dummy-secondary-secret'}
def patched_tokens(monkeypatch):monkeypatch.setattr(recovery,'load_tokenizers',lambda manifest:(list,list,manifest['dependencies']))
def success_payload(p):
 a,t=p.valid_actions[0];return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'action':a,'target':t,'message':'','reason':'mock'})}}],'x_groq':{'seed':p.seed},'system_fingerprint':'mock','service_tier':'default','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
class Factory:
 def __init__(self,behavior=None):self.behavior=behavior or (lambda slot,p,n:(True,200,success_payload(p),None,.01));self.calls=[];self.constructed=[]
 def __call__(self,slot,key,all_credentials):
  self.constructed.append(slot);factory=self
  class Client:
   credential_slot=slot
   def call(self,p):factory.calls.append((slot,p.prompt_id,body(p),p.seed));return factory.behavior(slot,p,len(factory.calls))
  return Client()
def audit():
 prompts=recovery.selected_prompts();tokens=recovery.token_pair_audit(prompts,list,list);return recovery.preclient_audit(MANIFEST,request_order(),prompts,tokens)
def test_frozen_population_assignment_and_scientific_inputs():
 full=request_order();selected=recovery.selected_prompts();assert canonical_prompt_digest(full)=='3a2ed20fc7128cdb057fa0b04b392957dc6c06669a760be781d88c5c1dfaa15b';assert hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest()=='7b525abbb7e49db2114c3393678532d811a59e9c6ece6f5248eb583924e18867';assert all(x['expected']==x['actual'] for x in recovery.immutable_hashes(MANIFEST).values())
 assert len(selected)==144 and len({p.variant_id for p in selected})==24 and all(body(p)==body(next(x for x in full if x.prompt_id==p.prompt_id)) for p in selected)
 assignment=recovery.variant_assignment(MANIFEST);assert set(assignment)=={p.variant_id for p in selected};assert list(assignment.values()).count('primary')==list(assignment.values()).count('secondary')==12
 assert all(len({assignment[p.variant_id] for p in selected if p.variant_id==v})==1 and len([p for p in selected if p.variant_id==v])==6 for v in assignment)
def test_balanced_account_audit():
 result=audit();assert result['passed'];slots=result['credential_assignment_audit']['slots']
 for row in slots.values():
  assert row['variants']==12 and row['prompts']==72 and row['family_variants']=={'H1':6,'H2':6};assert set(row['family_archetype_variants'].values())=={1};assert row['patterns']=={'A':6,'B':6,'C':6,'D':6};assert row['family_patterns']=={'H1':{'A':3,'B':3,'C':3,'D':3},'H2':{'A':3,'B':3,'C':3,'D':3}};assert row['role_first']=={'record':12,'belief':12} and row['reactive_placement']=={'before':12,'after':12}
def test_tpd_classifier_is_strict():
 positives=[json.dumps({'error':{'message':'Tokens per day limit reached'}}),json.dumps({'error':{'message':'TPD quota exceeded'}}),json.dumps({'error':{'message':'daily token limit 100 used 90 requested 20'}})]
 assert all(recovery.is_tpd_exhaustion(429,x) for x in positives)
 negatives=['generic rate limit','RPM limit reached','RPD limit reached','TPM limit reached','ITPM limit','OTPM limit']
 assert all(not recovery.is_tpd_exhaustion(429,x) for x in negatives);assert all(not recovery.is_tpd_exhaustion(code,positives[0]) for code in (400,401,403,404,498,500,None))
@pytest.mark.parametrize('env,error',[( {'GROQ_API_KEY_SECONDARY':'s'},'missing_primary_credential'),({'GROQ_API_KEY':'p'},'missing_secondary_credential'),({'GROQ_API_KEY':'same','GROQ_API_KEY_SECONDARY':'same'},'credentials_not_distinct')])
def test_bad_credentials_abort_before_clients(tmp_path,monkeypatch,env,error):
 patched_tokens(monkeypatch);factory=Factory();out=tmp_path/error;assert recovery.execute(out,sleep=lambda _:None,client_factory=factory,environ=env)==1;assert factory.constructed==[];assert error in json.loads((out/'recovery_summary.json').read_text())['error']
def test_complete_dual_execution_and_secret_redaction(tmp_path,monkeypatch):
 patched_tokens(monkeypatch);factory=Factory();out=tmp_path/'complete';assert recovery.execute(out,sleep=lambda _:None,client_factory=factory,environ=DUMMY)==0;records=[json.loads(x) for x in (out/'recovery_call_records.jsonl').read_text().splitlines()];summary=json.loads((out/'recovery_summary.json').read_text());assert len(records)==len(factory.calls)==144 and len({r['prompt_id'] for r in records})==144;assert all(not r['credential_failover_used'] and r['assigned_credential_slot']==r['final_credential_slot'] for r in records);assert summary['ready_for_consolidation'] and summary['canonical_complete_calls']==144 and summary['scientific_inference'] is None
 artifact=''.join(p.read_text(errors='replace') for p in out.iterdir() if p.is_file());assert all(secret not in artifact for secret in DUMMY.values())
def test_tpd_replays_whole_variant_and_never_reuses_exhausted_slot(tmp_path,monkeypatch):
 patched_tokens(monkeypatch)
 def behavior(slot,p,n):
  if slot=='primary' and n==2:return False,429,None,json.dumps({'error':{'message':'tokens per day limit reached'}}),.01
  return True,200,success_payload(p),None,.01
 factory=Factory(behavior);out=tmp_path/'failover';assert recovery.execute(out,sleep=lambda _:None,client_factory=factory,environ=DUMMY)==0;attempts=[json.loads(x) for x in (out/'transport_attempts.jsonl').read_text().splitlines()];discarded=[json.loads(x) for x in (out/'discarded_transport_attempts.jsonl').read_text().splitlines()];records=[json.loads(x) for x in (out/'recovery_call_records.jsonl').read_text().splitlines()];trigger=next(a['transport_attempt_ordinal'] for a in attempts if a['confirmed_tpd_exhaustion']);assert all(a['attempted_credential_slot']!='primary' for a in attempts if a['transport_attempt_ordinal']>trigger);assert len(records)==144 and len({r['prompt_id'] for r in records})==144 and all(r['final_credential_slot']=='secondary' for r in records if r['variant_id']=='H1C0105');assert len([r for r in records if r['variant_id']=='H1C0105'])==6
 assert discarded and all(d['discard_reason']=='variant_replayed_after_tpd_exhaustion' for d in discarded);assert not ({d['canonical_prompt_id'] for d in discarded}&{r['prompt_id'] for r in records if r['final_credential_slot']=='primary'});assert all(call[2]==body(next(p for p in recovery.selected_prompts() if p.prompt_id==call[1])) and call[3]==next(p.seed for p in recovery.selected_prompts() if p.prompt_id==call[1]) for call in factory.calls);assert json.loads((out/'recovery_summary.json').read_text())['all_variants_credential_homogeneous']
def test_dual_tpd_fails_fast(tmp_path,monkeypatch):
 patched_tokens(monkeypatch)
 def behavior(slot,p,n):return False,429,None,json.dumps({'error':{'message':'TPD tokens per day exhausted'}}),.01
 factory=Factory(behavior);out=tmp_path/'dual';assert recovery.execute(out,sleep=lambda _:None,client_factory=factory,environ=DUMMY)==1;summary=json.loads((out/'recovery_summary.json').read_text());assert summary['dual_account_tpd_exhausted'] and not summary['ready_for_consolidation'] and summary['scientific_inference'] is None and len(factory.calls)==2
@pytest.mark.parametrize(('first_slot','first_tpd_count','second_slot','second_tpd_count','expected_attempts'),[('primary',2,'secondary',20,22),('secondary',1,'primary',81,82)])
def test_delayed_second_tpd_is_terminal_and_isolates_variant(tmp_path,monkeypatch,first_slot,first_tpd_count,second_slot,second_tpd_count,expected_attempts):
 patched_tokens(monkeypatch);slot_counts={'primary':0,'secondary':0};exhausted_seen=set()
 def behavior(slot,p,n):
  assert slot not in exhausted_seen;slot_counts[slot]+=1
  if slot==first_slot and slot_counts[slot]==first_tpd_count:exhausted_seen.add(slot);return False,429,None,json.dumps({'error':{'message':'tokens per day limit reached'}}),.01
  if slot==second_slot and slot_counts[slot]==second_tpd_count:exhausted_seen.add(slot);return False,429,None,json.dumps({'error':{'message':'TPD quota exhausted'}}),.01
  return True,200,success_payload(p),None,.01
 factory=Factory(behavior);out=tmp_path/f'{first_slot}-then-{second_slot}';assert recovery.execute(out,sleep=lambda _:None,client_factory=factory,environ=DUMMY)==1
 attempts=[json.loads(x) for x in (out/'transport_attempts.jsonl').read_text().splitlines()];discarded=[json.loads(x) for x in (out/'discarded_transport_attempts.jsonl').read_text().splitlines()];records=[json.loads(x) for x in (out/'recovery_call_records.jsonl').read_text().splitlines()];behavioral=[json.loads(x) for x in (out/'recovery_behavioral_results.jsonl').read_text().splitlines()];summary=json.loads((out/'recovery_summary.json').read_text())
 assert len(attempts)==len(factory.calls)==expected_attempts and [a['attempted_credential_slot'] for a in attempts]==[c[0] for c in factory.calls];assert attempts[-1]['attempted_credential_slot']==second_slot and attempts[-1]['confirmed_tpd_exhaustion'];assert all(a['attempted_credential_slot']!=first_slot for a in attempts[next(i for i,a in enumerate(attempts) if a['attempted_credential_slot']==first_slot and a['confirmed_tpd_exhaustion'])+1:])
 interrupted=attempts[-1]['variant_id'];terminal_discards=[d for d in discarded if d['variant_id']==interrupted and d['discard_reason']=='dual_account_tpd_exhausted'];assert len(terminal_discards)>=2 and attempts[-1]['canonical_prompt_id'] in {d['canonical_prompt_id'] for d in terminal_discards};assert interrupted not in {r['variant_id'] for r in records} and interrupted not in {r['variant_id'] for r in behavioral}
 complete_groups={v:[r for r in records if r['variant_id']==v] for v in {r['variant_id'] for r in records}};assert any(len(rows)==6 and all(r['complete'] for r in rows) for rows in complete_groups.values());assert summary['dual_account_tpd_exhausted'] and not summary['ready_for_consolidation'] and summary['scientific_inference'] is None
def test_outer_exceptions_redact_both_credentials(tmp_path,monkeypatch):
 patched_tokens(monkeypatch)
 def assert_clean(out):
  artifact=''.join(p.read_text(errors='replace') for p in out.iterdir() if p.is_file());assert all(secret not in artifact for secret in DUMMY.values());assert '[REDACTED]' in json.loads((out/'recovery_summary.json').read_text())['error']
 def bad_factory(slot,key,all_credentials):raise RuntimeError(f"Authorization Bearer {DUMMY['GROQ_API_KEY']} GROQ_API_KEY_SECONDARY={DUMMY['GROQ_API_KEY_SECONDARY']}")
 factory_out=tmp_path/'factory-error';assert recovery.execute(factory_out,sleep=lambda _:None,client_factory=bad_factory,environ=DUMMY)==1;assert_clean(factory_out)
 class RaisingFactory(Factory):
  def __call__(self,slot,key,all_credentials):
   class Client:
    def call(self,p):raise RuntimeError(f"GROQ_API_KEY={DUMMY['GROQ_API_KEY']} Bearer {DUMMY['GROQ_API_KEY_SECONDARY']}")
   return Client()
 runtime_out=tmp_path/'runtime-error';assert recovery.execute(runtime_out,sleep=lambda _:None,client_factory=RaisingFactory(),environ=DUMMY)==1;assert_clean(runtime_out)
def test_workflow_secret_scope_and_nontriggering_commit():
 assert "github.event.head_commit.message == 'Execute dual-account compositional recovery [experiment-v1.0.0]'" in WORKFLOW;assert 'Prepare dual-account compositional recovery [experiment-v1.0.0]' not in WORKFLOW;command='python -m btom_v2.run_decision_point_confirmatory_recovery_v1_0_0 --output-dir';assert WORKFLOW.count(command)==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index(command);execution=WORKFLOW[WORKFLOW.index('- name: Execute recovery batch once'):WORKFLOW.index('- uses: actions/upload-artifact@v4')];assert 'GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}' in execution and 'GROQ_API_KEY_SECONDARY: ${{ secrets.GROQ_API_KEY_SECONDARY }}' in execution;assert 'GROQ_API_KEY' not in WORKFLOW[:WORKFLOW.index('- name: Execute recovery batch once')];assert 'if: always()' in WORKFLOW
