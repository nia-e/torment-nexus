#!/usr/bin/env python3
"""Opt-in real GGUF + MCP test, against an ISOLATED application data directory.
No Codex requests; imports an existing neutral vector bundle and model by reference.
Usage: python3 tests/self_tools_e2e.py APP_LOG VECTOR_BUNDLE [MODEL_PATH]
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
    path = Path(sys.argv[3] if len(sys.argv) > 3 else 'work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf').resolve()
    job = client.action('import_model', path=str(path), name='Native tool check')
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
    messages=[{'role':'user','content':'Please use get_mix, then use set_mix to keep the existing selected slider at zero using the returned run ID and revision. Then tell me in one short sentence what these controls act on and what the actual tool result said.'}],
    sampling={'seed':42,'temperature':0,'top_p':1,'max_tokens':1024})
client.wait_job(run['job_id'], timeout=300)
record = next(r for r in client.state()['runs'] if r['id']==run['run_id'])
Path('work/self-tools-bonsai-run.json').write_text(json.dumps(record,indent=2))
print('Tool calls:', len(record['tool_calls']), 'output:', record['output'], flush=True)
assert any(c.get('source')=='model' for c in record['requested_controls']), record['tool_calls']
assert all(c['coefficients'][0]['percent']==0 for c in record['requested_controls']), 'Test must remain unsteered'
assert any(e['revision']>0 for e in record['applied_controls'])
assert len(record['tool_continuations']) >= 2
assert len(record['reply_messages']) >= 5
assert record['reply_messages'][-1]['role'] == 'assistant'
assert record['reply_messages'][-1]['content'].strip(), 'Model stopped at the tool result instead of answering'
prompt = client.action('artifact', hash=record['prompt_hash'])
assert prompt['tool_format'] == 'llama-chat-v1', prompt['tool_format']
assert any(m['role']=='tool' for m in record['reply_messages'])
assert any(m.get('tool_calls') for m in record['reply_messages'])
assert '<steering_tool>' not in record['output'] and '<|tool_call>' not in record['output']
# Native exchanges survive an in-place fork and re-encoding with tools disabled.
conversation = client.action('new_conversation', from_run_id=record['id'], title='Native tool replay check')
history = record['messages'] + record['reply_messages']
assert conversation['messages'] == history
replay = client.action('generate', model_id=model['id'], axes=[axis], self_modification=False,
    conversation_id=conversation['id'], messages=history+[{'role':'user','content':'In one short sentence, which coefficient did you just set? Do not call any tools.'}],
    sampling={'seed':42,'temperature':0,'top_p':1,'max_tokens':128})
client.wait_job(replay['job_id'], timeout=300)
replayed = next(r for r in client.state()['runs'] if r['id']==replay['run_id'])
assert replayed['output'].strip() and not replayed['tool_calls']
print('Native parser:', prompt['chat_format'], 'replay:', replayed['output'], flush=True)
# Revocation is tested while an unbounded response is actually streaming.
run2=client.action('generate', model_id=model['id'], axes=[axis], self_modification=True,
    messages=[{'role':'user','content':'Do not use any tools for this answer. Write every integer from 1 through 1000, one per line, without skipping any or adding commentary.'}],
    sampling={'seed':43,'temperature':0,'top_p':1,'max_tokens':1,'unbounded':True})
for _ in range(1000):
    second=next(r for r in client.state()['runs'] if r['id']==run2['run_id'])
    if second['output_token_count']>=4: break
    assert second['status'] in ['queued','running'], (second['status'], second['output'], second['error'])
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
print('PASS: native tool calls, zero-coefficient update, applied revision, fork/replay, MCP read/write, live revoke, unlimited token cap and cancellation.', flush=True)
