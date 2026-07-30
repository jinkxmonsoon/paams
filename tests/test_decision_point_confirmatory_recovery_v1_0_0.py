import hashlib,json
from pathlib import Path
from btom_v2 import run_decision_point_confirmatory_recovery_v1_0_0 as recovery
from btom_v2.decision_point_confirmatory_prompting_v1_0_0 import canonical_prompt_digest,request_order
from btom_v2.run_decision_point_confirmatory_v1_0_0 import body
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_confirmatory_recovery_manifest_v1_0_0.json').read_text())
WORKFLOW=(ROOT/'.github/workflows/decision_point_confirmatory_recovery_v1_0_0.yml').read_text()
def audit_with_character_tokens():
 full=request_order();selected=recovery.selected_prompts();tokens=recovery.token_pair_audit(selected,list,list);return recovery.preclient_audit(MANIFEST,full,selected,tokens),tokens
def test_frozen_inputs_and_exact_population():
 full=request_order();selected=recovery.selected_prompts();assert canonical_prompt_digest(full)=='3a2ed20fc7128cdb057fa0b04b392957dc6c06669a760be781d88c5c1dfaa15b'
 assert hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest()=='7b525abbb7e49db2114c3393678532d811a59e9c6ece6f5248eb583924e18867'
 assert len(selected)==144 and len({p.prompt_id for p in selected})==144 and {p.difficulty for p in selected}=={'compositional'}
 assert {f:len({p.variant_id for p in selected if p.family==f}) for f in ('H1','H2')}=={'H1':12,'H2':12}
 assert [p.prompt_id for p in selected]==[p.prompt_id for p in full if p.difficulty=='compositional']
 assert all(v['expected']==v['actual'] for v in recovery.immutable_hashes(MANIFEST).values())
def test_manifest_identity_order_seed_and_body():
 full={p.prompt_id:p for p in request_order()};selected=recovery.selected_prompts();rows=MANIFEST['frozen_selected_order'];assert len(rows)==len(selected)
 for p,row in zip(selected,rows):
  original=full[p.prompt_id];assert row['prompt_id']==p.prompt_id and row['seed']==p.seed==original.seed and p.prompt==original.prompt and p.valid_actions==original.valid_actions and row['prompt_sha256']==hashlib.sha256(p.prompt.encode()).hexdigest();assert body(p)==body(original)
 assert all(a['original_ordinal']<b['original_ordinal'] for a,b in zip(rows,rows[1:]))
def test_triplets_patterns_positions_and_token_pairs():
 audit,tokens=audit_with_character_tokens();order=audit['order_audit'];assert audit['passed'];assert order['global_pattern_counts']=={'A':12,'B':12,'C':12,'D':12}
 assert len(order['triplets'])==48 and len({r['seed'] for r in order['triplets']})==48 and all(r['same_seed'] and r['role_pair_adjacent'] for r in order['triplets'])
 for row in order['family_state_strata'].values():
  assert row['pattern_counts']=={'A':3,'B':3,'C':3,'D':3} and row['role_first']=={'record':6,'belief':6} and row['reactive_placement']=={'before':6,'after':6}
  assert row['positions']=={'record':{1:3,2:6,3:3},'belief':{1:3,2:6,3:3}}
 assert tokens['pair_count']==48 and tokens['all_raw_parity'] and tokens['all_harmony_parity'] and tokens['all_content_matched']
def test_audit_is_preclient_and_configuration_is_frozen():
 source=(ROOT/'btom_v2/run_decision_point_confirmatory_recovery_v1_0_0.py').read_text();assert source.index('preclient_audit(manifest,full,prompts,tokens,client_constructed=False)')<source.index('client=client_factory()')
 audit,_=audit_with_character_tokens();assert audit['checks']['client_not_constructed'] and audit['checks']['request_body_identity']
 assert recovery.MAX_REQUESTS==144 and recovery.DELAY_SECONDS==20 and recovery.RETRIES==0 and recovery.COLLECTION_BATCH=='compositional_recovery_1'
def test_policy_workflow_and_no_scientific_inference(tmp_path,monkeypatch):
 policy=MANIFEST['recovery_policy'];assert policy['attempts_1_and_2_excluded'] and policy['previous_compositional_calls_discarded'] and not policy['cell_level_backfilling'] and not policy['cross_attempt_variant_assembly']
 class MockClient:
  calls=0
  def call(self,p):
   MockClient.calls+=1;a,t=p.valid_actions[0];content=json.dumps({'action':a,'target':t,'message':'','reason':'mock'});return True,200,{'choices':[{'finish_reason':'stop','message':{'content':content}}],'x_groq':{'seed':p.seed},'system_fingerprint':'mock-fp','service_tier':'default','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}},None,0.01
 monkeypatch.setattr(recovery,'load_tokenizers',lambda manifest:(list,list,manifest['dependencies']));out=tmp_path/'recovery';assert recovery.execute(out,sleep=lambda _:None,client_factory=MockClient)==0;assert MockClient.calls==144
 summary=json.loads((out/'recovery_summary.json').read_text());assert summary['attempted_requests']==144 and summary['scientific_inference'] is None and summary['final_confirmatory_analysis_not_performed'] and summary['previous_compositional_observations_excluded']
 assert all(json.loads(line)['collection_batch']=='compositional_recovery_1' for line in (out/'recovery_call_records.jsonl').read_text().splitlines())
 command='python -m btom_v2.run_decision_point_confirmatory_recovery_v1_0_0 --output-dir';assert WORKFLOW.count(command)==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index(command)
 assert "github.event.head_commit.message == 'Run complete compositional recovery batch [experiment-v1.0.0]'" in WORKFLOW and 'if: always()' in WORKFLOW and 'retention-days: 30' in WORKFLOW and WORKFLOW.count('GROQ_API_KEY')==2
