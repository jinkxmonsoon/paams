import hashlib,json
from pathlib import Path
import pytest
from btom_v2 import run_journal_j1_record_holdout_v0_6_2 as run
from btom_v2 import journal_j1_record_holdout_prompting_v0_6_1 as freeze
MANIFEST=json.loads(run.MANIFEST_PATH.read_text())

def validation(tmp_path,monkeypatch):
 d=tmp_path/'validation'; freeze.generate(d); (d/'workflow_status.json').write_text(json.dumps({'harmony_validation_status':'passed_complete_conversation_four_cell_parity','holdout_execution_ready':True}))
 names=MANIFEST['validated_v0_6_1_hashes']; hashes={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in names if (d/n).exists()}
 for n in names:
  if n not in hashes: (d/n).write_text(n); hashes[n]=hashlib.sha256((d/n).read_bytes()).hexdigest()
 m={**MANIFEST,'validated_v0_6_1_hashes':hashes}; p=tmp_path/'manifest.json'; p.write_text(json.dumps(m)); monkeypatch.setattr(run,'MANIFEST_PATH',p); return d,m
class Client:
 def __init__(self,secret,fail=False): self.secret=secret; self.calls=[]; self.fail=fail
 def call(self,prompt,seed):
  self.calls.append((prompt,seed))
  if self.fail:return False,500,None,'error',.1
  return True,200,{'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'action':'SEND_CORRECTION' if seed%2 else 'CONTINUE_TASK'})}}],'usage':{'total_tokens':1}},None,.1

def test_manifest_freeze_and_api_transport():
 assert MANIFEST['model']=='openai/gpt-oss-20b' and MANIFEST['maximum_behavioral_requests']==72 and len(MANIFEST['frozen_seeds'])==72 and len(MANIFEST['frozen_request_order'])==72
 body=run.request_body('x',1); assert body==run.base.request_body('x',1); assert body['messages']==[{'role':'user','content':'x'}] and 'tools' not in body and 'rationale' not in json.dumps(body).lower()
 assert run.USER_AGENT=='Mozilla/5.0 (compatible; BToM-MAS/1.0.0; +https://github.com/jinkxmonsoon/paams)'
 assert run.Client('secret').headers()['User-Agent']==run.USER_AGENT

def test_hash_and_harmony_gate_before_client(tmp_path,monkeypatch):
 d,m=validation(tmp_path,monkeypatch); (d/next(iter(m['validated_v0_6_1_hashes']))).write_text('bad'); made=[]
 assert run.execute(tmp_path/'out',d,client_factory=lambda key:made.append(key),environ={'GROQ_API_KEY':'secret'})==1 and not made

def test_success_72_preflight_retained_delay_and_secret_redaction(tmp_path,monkeypatch):
 d,_=validation(tmp_path,monkeypatch); c=Client('secret'); sleeps=[]
 assert run.execute(tmp_path/'out',d,sleep=sleeps.append,client_factory=lambda secret:c,environ={'GROQ_API_KEY':'secret'})==0
 assert len(c.calls)==72 and len(sleeps)==71 and sleeps==[20]*71
 summary=json.loads((tmp_path/'out/workflow_status.json').read_text()); assert summary['attempted_calls']==summary['complete_calls']==72 and summary['technical_pass'] and summary['fallback_actions']==0
 assert 'secret' not in ''.join(p.read_text() for p in (tmp_path/'out').iterdir())

def test_failed_preflight_stops_one_and_no_gates(tmp_path,monkeypatch):
 d,_=validation(tmp_path,monkeypatch); c=Client('secret',True)
 assert run.execute(tmp_path/'out',d,sleep=lambda _:None,client_factory=lambda _:c,environ={'GROQ_API_KEY':'secret'})==1 and len(c.calls)==1
 s=json.loads((tmp_path/'out/workflow_status.json').read_text()); assert s['attempted_calls']==1 and s['classification']=='technical_failure' and not s['scenario_control_pass']

def test_client_factory_secret_is_redacted(tmp_path,monkeypatch):
 d,_=validation(tmp_path,monkeypatch); secret='dummy-j1-secret-value'
 def factory(_): raise RuntimeError(secret)
 assert run.execute(tmp_path/'out',d,client_factory=factory,environ={'GROQ_API_KEY':secret})==1
 assert secret not in ''.join(p.read_text() for p in (tmp_path/'out').iterdir())

def rows(c00,c01,c10,c11):
 out=[]
 for mr,mi,value in ((0,0,c00),(0,1,c01),(1,0,c10),(1,1,c11)):
  for i in range(18): out.append({'M_R':mr,'M_I':mi,'action':'SEND_CORRECTION' if i<value*18 else 'CONTINUE_TASK','complete':True,'archetype':f'a{i%6}','difficulty':f'd{i%3}'})
 return out
@pytest.mark.parametrize('rates,classification',[((0,0,0,0),'record_holdout_relevance_failure'),((0,0,1,1),'causal_control_valid_but_measurement_saturated'),((0,.5,.5,1),'generic_mismatch_salience_replication'),((1/3,1/3,2/3,2/3),'record_holdout_validated')])
def test_metrics_and_classification(rates,classification):
 result=run.metrics(rows(*rates),True); assert result['classification']==classification
 assert result['record_relevance_pooled']==pytest.approx((rates[2]+rates[3]-rates[0]-rates[1])/2)
 assert result['record_relevance_MI0']==pytest.approx(rates[2]-rates[0]) and result['record_relevance_MI1']==pytest.approx(rates[3]-rates[1])
 assert result['record_irrelevant_MR0']==pytest.approx(rates[1]-rates[0]) and result['record_irrelevant_MR1']==pytest.approx(rates[3]-rates[2])

def test_strict_parse_no_repair_fallback_or_120b():
 assert run.strict_parse('{"action":"SEND_CORRECTION"}')=='SEND_CORRECTION'; assert run.strict_parse('text {"action":"SEND_CORRECTION"}') is None
 source=Path(run.__file__).read_text(); assert 'gpt-oss-120b' not in source and 'confirmatory_prompts_generated' in source
