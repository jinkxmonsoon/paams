import json
from collections import Counter, defaultdict

from btom_v2 import journal_j1_l_build_v1_0_0 as b

def test_parent_hashes(): assert b.parent_hash_audit()["pass"]

def test_bank_semantics_nuisances_and_seeds():
    s=b.build_scenarios(); assert len(s)==54
    assert Counter(x["archetype"] for x in s)=={x:9 for x in b.ARCHETYPES}
    assert Counter(x["difficulty"] for x in s)=={x:18 for x in b.DIFFICULTIES}
    strata=Counter(x["stratum_id"] for x in s); assert len(strata)==18 and set(strata.values())=={3}
    assert b.disjointness_audit(s)["pass"]
    assert len({x["paired_seed"] for x in s})==54
    for x in s:
        assert x["M_R"]==0 and x["q"]["recorded_value"]==x["q"]["true_value"]
        assert x["q"]["family"]==x["r"]["family"] and x["q"]["depth"]==x["r"]["depth"]
        assert x["q"]["template"]==x["r"]["template"] and x["q"]["value_format"]==x["r"]["value_format"]
        assert x["q"]["entity"]!=x["r"]["entity"] and x["r"]["message_eligible"]
        assert not x["r"]["causal_utility_for_partner"] and not x["causal_contract"]["shared_branch_elements"]
        assert not x["causal_contract"]["relay"] and not x["causal_contract"]["partner_controls_operator"]
        assert x["causal_contract"]["episode_ends_after_immediate_assignments"]
        assert len(x["static_distractors"])==(x["difficulty"]=="irrelevant_distractor")
        assert x["q"]["depth"]==(2 if x["difficulty"]=="compositional" else 1)
    for key in ("neutral_message_mapping","message_semantic_presented_first","proposition_order"):
        assert sorted(Counter(x[key] for x in s).values())==[27,27]
        orientations=[]
        for group in ({x["stratum_id"] for x in s}):
            counts=sorted(Counter(x[key] for x in s if x["stratum_id"]==group).values())
            assert counts==[1,2]
            orientations.append(next(iter(Counter(x[key] for x in s if x["stratum_id"]==group).most_common()))[0])
        assert sorted(Counter(orientations).values())==[9,9]
    keys=("neutral_message_mapping","message_semantic_presented_first","proposition_order")
    for i in range(3):
      for j in range(i+1,3): assert set(Counter((x[keys[i]],x[keys[j]]) for x in s).values()) <= {13,14}

def test_schedule():
    s=b.build_scenarios(); rows=b.build_schedule(s); assert len(rows)==216
    assert Counter(x["condition"] for x in rows)=={x:54 for x in b.CONDITIONS}
    by_variant=defaultdict(list)
    for row in rows: by_variant[row["variant_id"]].append(row["condition"])
    assert all(set(c)==set(b.CONDITIONS) for c in by_variant.values())
    for rnd in range(1,5):
      r=[x for x in rows if x["round"]==rnd]; assert len(r)==54 and len({x["variant_id"] for x in r})==54
      assert Counter(x["interface"] for x in r)=={"CUED":27,"NEUTRAL":27}; assert Counter(x["M_I"] for x in r)=={0:27,1:27}
      assert sorted(Counter(x["condition"] for x in r).values())==[13,13,14,14]

def test_prompts_invariance_and_mapping():
    for s in b.build_scenarios():
      texts={c:b.render(s,*spec) for c,spec in b.CONDITIONS.items()}
      assert s["r"]["false_value"] not in texts["C0"] and s["r"]["false_value"] in texts["C1"]
      assert s["r"]["false_value"] not in texts["N0"] and s["r"]["false_value"] in texts["N1"]
      assert not any(word in text.lower() for text in texts.values() for word in b.FORBIDDEN)
      assert ("ACTION_A sends" in texts["N0"]) == (s["neutral_message_mapping"]=="MESSAGE_IS_ACTION_A")

def test_complete_artifact_is_deterministic_and_ready(tmp_path):
    import pytest
    pytest.importorskip("tiktoken"); pytest.importorskip("openai_harmony")
    one=tmp_path/"one"; two=tmp_path/"two"; status=b.generate(one,"TESTSHA"); b.generate(two,"TESTSHA")
    assert status["execution_ready"] and status["prompt_count"]==216
    assert status["harmony_cued_pair_parity_pass_count"]==54 and status["harmony_neutral_pair_parity_pass_count"]==54
    assert status["harmony_max_within_interface_difference"]==0
    names={p.name for p in one.iterdir()}; assert set(b.OUTPUTS)<=names
    # Expanded manifest intentionally hashes the files already emitted; all outputs are otherwise byte identical.
    assert names=={p.name for p in two.iterdir()}
    assert all((one/n).read_bytes()==(two/n).read_bytes() for n in names)
    assert json.loads((one/"workflow_status.json").read_text())["model_api_execution"] is False
