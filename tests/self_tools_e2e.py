#!/usr/bin/env python3
"""Opt-in real Bonsai + MCP test, against an ISOLATED application data directory.
No Codex requests; imports an existing neutral vector bundle and model by reference.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request
from app_e2e import Client

client = Client(sys.argv[1])
bundle = json.loads(Path(sys.argv[2]).read_text())
state = client.state()
assert not state['models'] or len(state['vectors']) <= 1, 'use the isolated test lab'
if not state['models']:
    job = client.action('import_model', path=str(Path('work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf').resolve()), name='Bonsai tool check')
    if job.get('job_id'): client.wait_job(job['job_id'])
if not client.state()['vectors']:
    client.action('import_vector', bundle=bundle)
state = client.state()
model = state['models'][0]
vector = state['vectors'][0]
layer = vector['selected_layer']
axis = {'vector_id': vector['id'], 'layer': layer, 'percent': 0}

def mcp(method, params=None):
    connection = client.state()['mcp']
    request = urllib.request.Request(connection['url'], json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params or {}}).encode(), {'Authorization':'Bearer '+connection['token'], 'Content-Type':'application/json','Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-06-18'})
    with urllib.request.urlopen(request, timeout=60) as response: return json.load(response)

assert mcp('initialize')['result']['protocolVersion'] == '2025-06-18'
assert {t['name'] for t in mcp('tools/list')['result']['tools']} == {'get_mix','set_mix'}
run = client.action('generate', model_id=model['id'], axes=[axis], self_modification=True,
    messages=[{'role':'user','content':'Please use get_mix, then use set_mix to set the existing selected slider to -1 percent using the returned run ID and revision. Then tell me in one short sentence what the actual tool result said. Do not merely describe a tool call: use the exact torment_tool envelope from your instructions.'}],
    sampling={'seed':42,'temperature':0,'top_p':1,'max_tokens':1024})
client.wait_job(run['job_id'], timeout=300)
record = next(r for r in client.state()['runs'] if r['id']==run['run_id'])
Path('work/self-tools-bonsai-run.json').write_text(json.dumps(record,indent=2))
print('Tool calls:', len(record['tool_calls']), 'output:', record['output'], flush=True)
assert any(c.get('source')=='model' and c['coefficients'][0]['percent']==-1 for c in record['requested_controls']), record['tool_calls']
assert any(e['revision']>0 for e in record['applied_controls'])
assert len(record['tool_continuations']) >= 2
assert len(record['reply_messages']) >= 5
assert record['reply_messages'][-1]['role'] == 'assistant'
assert record['reply_messages'][-1]['content'].strip(), 'Model stopped at the tool result instead of answering'
# Revocation is tested while an unbounded response is actually streaming.
run2=client.action('generate', model_id=model['id'], axes=[axis], self_modification=True,
    messages=[{'role':'user','content':'Do not use any tools for this answer. Count upward from one, writing each integer on its own line. Keep going until I stop you.'}],
    sampling={'seed':43,'temperature':0,'top_p':1,'max_tokens':1,'unbounded':True})
for _ in range(1000):
    second=next(r for r in client.state()['runs'] if r['id']==run2['run_id'])
    if second['output_token_count']>=30: break
    assert second['status'] in ['queued','running'], second
    time.sleep(.05)
assert second['output_token_count']>1
mix=mcp('tools/call',{'name':'get_mix','arguments':{}})['result']['structuredContent']
assert mix['run_id']==run2['run_id'] and 'messages' not in mix
changed=mcp('tools/call',{'name':'set_mix','arguments':{'run_id':mix['run_id'],'expected_revision':mix['revision'],'changes':[dict(axis,percent=0)],'reason':'MCP transport check, preserve the zero baseline'}})
assert changed['result']['isError'] is False, changed
client.action('set_self_modification',run_id=run2['run_id'],enabled=False)
assert mcp('tools/call',{'name':'get_mix','arguments':{}})['result']['isError'] is True
client.action('cancel_run',run_id=run2['run_id'])
for _ in range(200):
    second=next(r for r in client.state()['runs'] if r['id']==run2['run_id'])
    if second['status'] not in ['running','queued']: break
    time.sleep(.05)
assert second['status']=='cancelled', second
Path('work/self-tools-mcp-run.json').write_text(json.dumps(second,indent=2))
print('PASS: native Bonsai tool calls, signed change, applied revision, MCP read/write, live revoke, unlimited token cap and cancellation.', flush=True)
