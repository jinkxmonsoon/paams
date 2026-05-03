from __future__ import annotations
from dataclasses import asdict
from .env import BTomEnvV2
from .policies import SharedMemoryPolicy, SecondOrderBeliefPolicy

def run(seed, policy_cls):
    env=BTomEnvV2('C4_communication_delay',seed=seed,max_turns=40); p=policy_cls()
    for t in range(env.max_turns):
        for a in ('A','B','C'):
            act,tgt,meta=p.act(env,a,t); env.step(a,act,tgt,meta)
            if env.state.done: break
    s=asdict(env.summary()); s['policy']=policy_cls.name; return s

def main():
    rows=[]
    for pol in (SharedMemoryPolicy,SecondOrderBeliefPolicy):
        for seed in (0,1,2,3,4): rows.append(run(seed,pol))
    for r in rows: print(r)
    def group(pol):
        rs=[r for r in rows if r['policy']==pol]; n=len(rs)
        mean=lambda k: sum(r[k] for r in rs)/n
        return {"policy":pol,"N":n,"success_rate":mean('success'),"mean_turns":mean('turns'),"mean_messages":mean('messages_sent_count'),"mean_delayed_messages_count":mean('delayed_messages_count'),"mean_delivered_delayed_messages_count":mean('delivered_delayed_messages_count'),"mean_premature_shared_memory_assumptions":mean('premature_shared_memory_assumptions'),"mean_delayed_message_confusion_events":mean('delayed_message_confusion_events'),"mean_second_order_delivery_waits":mean('second_order_delivery_waits')}
    print('C4_GROUPED')
    g1=group('SharedMemoryPolicy'); g2=group('SecondOrderBeliefPolicy'); print(g1); print(g2)
    assert any(r['delayed_messages_count']>0 for r in rows)
    assert 'mean_second_order_delivery_waits' in g2
    assert 'mean_premature_shared_memory_assumptions' in g1
    assert g2['mean_messages'] < 10
    from .policies import SharedMemoryPolicy as SMP, SecondOrderBeliefPolicy as SOP
    assert SMP.uses_global_truth is False and SOP.uses_global_truth is False
    print('sanity_checks=PASS')

if __name__ == "__main__":
    main()
