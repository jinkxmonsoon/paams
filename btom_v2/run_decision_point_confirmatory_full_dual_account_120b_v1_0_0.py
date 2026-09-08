"""Thin adapter for the precommitted GPT-OSS 120B confirmatory replication."""
from __future__ import annotations
import argparse, json, os, time, urllib.error, urllib.request
from pathlib import Path
from . import run_decision_point_confirmatory_full_dual_account_v1_0_0 as engine
from .run_decision_point_confirmatory_v1_0_0 import USER_AGENT, body as frozen_body

MODEL='openai/gpt-oss-120b'
COLLECTION_BATCH='full_confirmatory_120b_dual_account_replication'
MANIFEST=Path(__file__).with_name('decision_point_confirmatory_full_dual_account_120b_manifest_v1_0_0.json')
ANALYZER_GIT_BLOB='15701d071f5fdb87209bf3247931e1558c7f8f92'
PROMPT_DIGEST='3a2ed20fc7128cdb057fa0b04b392957dc6c06669a760be781d88c5c1dfaa15b'
SCENARIO_HASH='7b525abbb7e49db2114c3393678532d811a59e9c6ece6f5248eb583924e18867'
PARENT_20B={'parent_20b_run_id':34036092185,'parent_20b_execution_sha':'e290e07b1a60ea12befb7f6c76a2fc7f0997a132','parent_20b_artifact_id':9992434005,'parent_20b_artifact_zip_sha256':'0d87972d686e7cbc54aca181bd390add0af5fee335afe927473d0894d10e332e'}

def request_body(prompt):
 payload=frozen_body(prompt).copy();payload['model']=MODEL;return payload

class ReplicationRoutedClient(engine.RoutedClient):
 def __init__(self,credential_slot,credential,all_credentials):
  super().__init__(credential_slot,credential,all_credentials);self.model=MODEL
 def call(self,prompt):
  request=urllib.request.Request(self.endpoint,data=json.dumps(request_body(prompt)).encode(),headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json','Accept':'application/json','User-Agent':USER_AGENT},method='POST');start=time.time()
  try:
   with urllib.request.urlopen(request,timeout=120) as response:return True,response.status,json.loads(response.read()),None,time.time()-start
  except urllib.error.HTTPError as exc:return False,exc.code,None,self._sanitize(exc.read().decode(errors='replace')),time.time()-start
  except Exception as exc:return False,None,None,self._sanitize(str(exc)),time.time()-start

def default_client_factory(slot,credential,all_credentials):
 return ReplicationRoutedClient(slot,credential,all_credentials)

def provenance(environ=None):
 env=os.environ if environ is None else environ
 return {'collection_batch':COLLECTION_BATCH,'model':MODEL,'actual_execution_sha':env.get('GITHUB_SHA') or None,**PARENT_20B,'analyzer_git_blob':ANALYZER_GIT_BLOB,'prompt_digest':PROMPT_DIGEST,'scenario_hash':SCENARIO_HASH,'replication_precommitted':True,'behavioral_20b_outputs_used_for_design':False}

def execute(out:Path,sleep=time.sleep,client_factory=default_client_factory,environ=None):
 old_manifest,old_batch=engine.MANIFEST,engine.COLLECTION_BATCH
 try:
  engine.MANIFEST=MANIFEST;engine.COLLECTION_BATCH=COLLECTION_BATCH
  result=engine.execute(out,sleep=sleep,client_factory=client_factory,environ=environ)
 finally:
  engine.MANIFEST,engine.COLLECTION_BATCH=old_manifest,old_batch
 if out.is_dir():engine.dump(out/'confirmatory_full_dual_replication_provenance.json',provenance(environ))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);return execute(parser.parse_args().output_dir)
if __name__=='__main__':raise SystemExit(main())
