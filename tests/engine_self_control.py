#!/usr/bin/env python3
"""Real Bonsai small-context rollover and cancellation while waiting on a tool."""
import json
from pathlib import Path
from engine_conformance import Worker
worker=Worker('engine/build/bin/torment-engine','work/self-bonsai-control-engine.log')
results={}
try:
    worker.call('load',path=str(Path('work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf').resolve()),context=256,gpu_layers=99,batch=128,microbatch=64)
    ident=worker.send('generate',messages=[{'role':'user','content':'Count upward from 1, writing each integer on its own line. Keep counting until stopped. Do not explain or conclude.'}],sampling={'max_tokens':1,'unbounded':True,'temperature':0,'seed':42,'top_p':1},controls={'revision':0,'rows':[]})
    rollovers=[];stop=False
    while True:
        event=worker.recv()
        if event['id']!=ident:continue
        if event['event']=='context_rollover':
            rollovers.append(event)
            print('Bonsai rolled at token',event['first_token_index'],flush=True)
            if not stop:worker.send('cancel',target=ident);stop=True
        if event['event']=='token' and event['index']>1200 and not stop:worker.send('cancel',target=ident);stop=True
        if event['event'] in ['done','error']:
            assert event['event']=='done' and event['cancelled'] and rollovers,event
            results['rollovers']=rollovers
            results['tokens']=event['tokens']
            break
    worker.call('unload')
    worker.call('load',path=str(Path('work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf').resolve()),context=1024,gpu_layers=99,batch=128,microbatch=64)
    ident=worker.send('generate',tools_enabled=True,messages=[{'role':'system','content':'You have a get_mix tool. To invoke it, emit exactly <torment_tool>{"name":"get_mix","arguments":{}}</torment_tool> and wait for its result.'},{'role':'user','content':'Use get_mix now.'}],sampling={'max_tokens':128,'temperature':0,'seed':42,'top_p':1},controls={'revision':0,'rows':[]})
    called=False
    while True:
        event=worker.recv()
        if event['id']!=ident:continue
        if event['event']=='tool_call':
            called=True
            worker.send('cancel',target=ident)
        if event['event'] in ['done','error']:
            assert event['event']=='done' and event['cancelled'] and called,event
            results['tool_wait_cancelled']=True
            break
    Path('work/self-bonsai-control-result.json').write_text(json.dumps(results,indent=2))
    print('PASS: Bonsai recurrent/attention context rollover, and Stop during a paused tool call.',flush=True)
finally:worker.close()
