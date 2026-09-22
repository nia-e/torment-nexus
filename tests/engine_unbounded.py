#!/usr/bin/env python3
"""Real small-model context rollover / Stop check, not a mock decoder."""
from pathlib import Path
import json
from engine_conformance import Worker
worker=Worker('engine/build/bin/torment-engine','work/self-unbounded-engine.log')
try:
    loaded=worker.call('load',path=str(Path('work/models/stories260K.gguf').resolve()),context=128,gpu_layers=0,batch=64,microbatch=32)
    ident=worker.send('generate',messages=[{'role':'user','content':'Once upon a time, there was a little girl named Lily. She loved to explore the forest. One day, she found'}],raw=True,sampling={'max_tokens':1,'unbounded':True,'temperature':0,'seed':42,'top_p':1},controls={'revision':0,'rows':[]})
    rolled=[]; count=0; stop=False
    while True:
        e=worker.recv()
        if e['id']!=ident: continue
        if e['event']=='context_rollover': rolled.append(e)
        if e['event']=='token':
            count=e['index']+1
            if (rolled or count>=1000) and not stop:
                worker.send('cancel',target=ident); stop=True
        if e['event'] in ['done','error']:
            assert e['event']=='done',e
            assert rolled and any(r['first_token_index']>0 for r in rolled),(count,e,rolled)
            assert e['cancelled'],e
            assert count>1,count
            Path('work/self-unbounded-result.json').write_text(json.dumps({'token_count':count,'rollovers':rolled,'cancelled':e['cancelled']},indent=2))
            print('PASS: cap=1 ignored, generated',count,'tokens, rolled context and cancelled.',flush=True)
            break
finally: worker.close()
