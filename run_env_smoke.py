from __future__ import annotations
from collections import defaultdict
from statistics import mean
from btom_env import BToMEnvironment
from policies import RandomPolicy, GreedySharedMemoryPolicy, GreedyBeliefStatePolicy, GreedySecondOrderBeliefPolicy

SCENARIOS=["C1_fully_observable","C2_partial_observable","C4_communication_delay","C5_false_belief_injection","C6_resource_allocation"]
POLICIES={"RandomPolicy":RandomPolicy,"GreedySharedMemoryPolicy":GreedySharedMemoryPolicy,"GreedyBeliefStatePolicy":GreedyBeliefStatePolicy,"GreedySecondOrderBeliefPolicy":GreedySecondOrderBeliefPolicy}
SEEDS=[0,1,2]

def main()->None:
    rows=[]
    for sc in SCENARIOS:
        for pn,pc in POLICIES.items():
            for seed in SEEDS:
                env=BToMEnvironment(); env.reset(seed,sc); p=pc(seed)
                ep=env.run_episode(lambda a,o,_e: p.act(a,o),30)
                rows.append({"scenario_id":sc,"policy":pn,"success":ep["success"],"turns":ep["turns_to_completion"],"messages":ep["total_messages"],"delayed_messages_count":ep["delayed_messages_count"],"false_belief_caused_wasted_action":ep["false_belief_caused_wasted_action"],"duplicate_resource_attempts":ep["duplicate_resource_attempts"],"wrong_allocation_attempts":ep["wrong_allocation_attempts"]})
    grp=defaultdict(list)
    for r in rows: grp[(r['scenario_id'],r['policy'])].append(r)
    print("scenario_id | policy | success_rate | mean_turns | mean_messages | delayed_messages_count | false_belief_caused_wasted_action | duplicate_resource_attempts | wrong_allocation_attempts")
    for (sc,pn),rr in sorted(grp.items()):
        print(f"{sc} | {pn} | {mean([1.0 if x['success'] else 0.0 for x in rr]):.2f} | {mean([x['turns'] for x in rr]):.2f} | {mean([x['messages'] for x in rr]):.2f} | {mean([x['delayed_messages_count'] for x in rr]):.2f} | {mean([x['false_belief_caused_wasted_action'] for x in rr]):.2f} | {mean([x['duplicate_resource_attempts'] for x in rr]):.2f} | {mean([x['wrong_allocation_attempts'] for x in rr]):.2f}")
    print("\nChecks:")
    for s in ["C1_fully_observable","C2_partial_observable"]:
        for p in ["GreedySharedMemoryPolicy","GreedyBeliefStatePolicy","GreedySecondOrderBeliefPolicy"]:
            m=mean([1.0 if x['success'] else 0.0 for x in grp[(s,p)]])
            print(f"- {s} {p} success_rate>0: {m>0} ({m:.2f})")
    c4_msgs=[x['messages'] for x in rows if x['scenario_id']=='C4_communication_delay' and 'Greedy' in x['policy']]
    print(f"- C4 greedy mean messages < 15: {mean(c4_msgs)<15} ({mean(c4_msgs):.2f})")

if __name__=='__main__': main()
