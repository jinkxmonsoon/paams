import ast,csv,json,math
from pathlib import Path
from btom_v2 import journal_j1_lexical_ablation_power_precision_v0_7_0 as p
ROOT=Path(__file__).resolve().parents[1]; BTOM=ROOT/'btom_v2'
PLAN=json.loads((BTOM/'journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json').read_text())
GRID=list(csv.DictReader((BTOM/'journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv').open()))

def brute(n,pp,pm):
 # Exhaustively enumerate four paired outcomes for small n.
 probs=(1-pp-pm,pp,pm,0.0); total=0
 import itertools
 for outcomes in itertools.product(range(4),repeat=n):
  probability=math.prod(probs[x] for x in outcomes)
  plus=outcomes.count(1); minus=outcomes.count(2); d=plus+minus; threshold=p.rejection_threshold(d)
  if threshold is not None and plus>=threshold: total+=probability
 return total

def test_frozen_grids_and_formula():
 assert p.CANDIDATE_N==(18,24,36,54,72) and p.P_MINUS==(0,.025,.05,.10)
 assert p.DELTA==.25 and p.ALPHA==.05 and p.MIN_POWER==.80 and len(GRID)==20
 for r in GRID: assert float(r['p_plus'])==float(r['p_minus'])+.25 and float(r['p_plus'])+float(r['p_minus'])<=1 and 0<=float(r['exact_one_sided_mcnemar_power'])<=1

def test_threshold_edges_and_bruteforce():
 assert p.rejection_threshold(0) is None and p.rejection_threshold(4) is None and p.rejection_threshold(5)==5 and p.rejection_threshold(6)==6
 assert p.exact_power(4,.4,.1)==pytest.approx(brute(4,.4,.1))
 assert p.exact_power(5,.25,0)==pytest.approx(brute(5,.25,0))

def test_selection_balance_and_no_skipping():
 summaries={int(k):v for k,v in PLAN['candidate_summary'].items()}; assert summaries[24]['exact_strata_balance_eligible'] is False
 for n in (18,36,54,72): assert summaries[n]['variants_per_stratum']==n//18
 eligible=[n for n in p.CANDIDATE_N if summaries[n]['scientifically_power_eligible']]
 assert PLAN['selected_N']==min(eligible)==54 and all(not summaries[n]['scientifically_power_eligible'] for n in p.CANDIDATE_N if n<54)
 assert PLAN['selected_N']%2==0 and PLAN['selected_variants_per_stratum']==3

def test_operational_calculations_and_budget_rule(monkeypatch):
 for n in p.CANDIDATE_N:
  r=next(x for x in GRID if int(x['N'])==n); assert int(r['total_calls'])==4*n and int(r['inter_call_delays'])==4*n-1 and int(r['delay_only_seconds'])==(4*n-1)*20
 assert PLAN['selected_total_calls']==216 and PLAN['planning_status']=='sample_size_selected' and PLAN['selected_single_run_delay_budget_feasible']
 monkeypatch.setattr(p,'TIMEOUT_MINUTES',1); assert p.plan()['selected_N']==54 and p.plan()['planning_status']=='scientific_sample_size_requires_execution_budget_revision'

def test_historical_effect_not_used_and_lambda_is_not_power():
 source=Path(p.__file__).read_text(); assert '12/18' not in source.split('def plan')[0]
 assert PLAN['historical_effect_used_for_power_selection'] is False and PLAN['lambda_precision']['mean_se_upper_bound_formula']=='2/sqrt(N), conservative descriptive bound'

def test_no_outcomes_network_secrets_or_generation():
 source=Path(p.__file__).read_text(); tree=ast.parse(source); imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
 assert not imports & {'requests','httpx','openai','groq','socket','urllib','numpy','scipy'}
 assert not any(x in source for x in ('GROQ_API_KEY','behavioral_results','raw_api','subgroup','record_holdout_call_records'))
 assert PLAN['behavioral_observations']==0 and not PLAN['model_api_execution'] and not PLAN['final_scenarios_generated'] and not PLAN['final_prompts_generated']

def test_design_required_freezes():
 text=(BTOM/'journal_j1_lexical_ablation_design_v0_7_0.md').read_text()
 for phrase in ('variant is the inferential unit','`M_R` is fixed at zero','C0, C1, N0, and N1','mapping is counterbalanced','sole primary test is neutral N1 versus N0','key secondary','No Lambda equivalence threshold is frozen','must be new and disjoint','120B is deferred','ICAART remains frozen'):
  assert phrase in text

def test_deterministic_regeneration(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';p.generate(a);p.generate(b)
 for name in ('journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json','journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv'): assert (a/name).read_bytes()==(b/name).read_bytes()==(BTOM/name).read_bytes()

import pytest
