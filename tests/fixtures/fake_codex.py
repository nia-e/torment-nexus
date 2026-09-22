#!/usr/bin/python3
# Test-only app-server. No network, no real Codex auth or configuration access.
import json, os, sys, time
SCENARIO = __SCENARIO__
DISABLED = __DISABLED__
if sys.argv[1:4] == ['debug', 'models', '--bundled']:
    print(json.dumps({'models':[{'slug':'fake-model'}]})); sys.exit(0)
options = {}
for i, arg in enumerate(sys.argv):
    if arg == '-c':
        key, value = sys.argv[i+1].split('=',1)
        try: value = json.loads(value)
        except Exception: pass
        options[key] = value
counter_path = os.path.join(os.path.dirname(__file__), 'counter')
requests_path = os.path.join(os.path.dirname(__file__), 'requests.jsonl')
recovered_path = os.path.join(os.path.dirname(__file__), 'auth-recovered')
def send(value):
    print(json.dumps(value), flush=True)
def event(method, params): send({'method':method,'params':params})
def completed(status='completed'):
    event('turn/completed',{'threadId':'thread','turn':{'id':'turn','status':status,'error':None}})
for line in sys.stdin:
    request=json.loads(line); method=request.get('method'); ident=request.get('id')
    with open(requests_path, 'a') as log:
        log.write(json.dumps({'method':method})+'\n')
    if ident is None: continue
    result={}
    if method=='initialize': result={'userAgent':'torment_nexus/0.155.1 (test)'}
    elif method=='config/read':
        features={k:False for k in DISABLED};features['skip_host_skill_discovery']=True
        result={'config':{'features':features,'mcp_servers':{},'web_search':'disabled','memories':{'use_memories':False,'generate_memories':False},'notify':[],'skills':{'include_instructions':False},'model_catalog_json':options.get('model_catalog_json')}}
    elif method=='configRequirements/read': result={'requirements':None}
    elif method=='experimentalFeature/list': result={'data':[{'name':k,'enabled':False} for k in DISABLED],'nextCursor':None}
    elif method=='account/read': result={'account':None if SCENARIO=='signed_out' else {'type':'chatgpt'}}
    elif method=='model/list':
        result={'data':[{'id':'fake-model','model':'fake-model','displayName':'Fake','isDefault':True,'inputModalities':['text']}],'nextCursor':None}
        if SCENARIO == 'catalog_mismatch':
            result['data'].insert(0, {'id':'uninstalled-model','model':'uninstalled-model','displayName':'Unsupported','isDefault':True,'inputModalities':['text']})
    elif method=='thread/start': result={'thread':{'id':'thread'},'sandbox':{'type':'readOnly','networkAccess':False},'approvalPolicy':'on-request','approvalsReviewer':'auto_review'}
    elif method=='mcpServerStatus/list': result={'data':[],'nextCursor':None}
    elif method=='app/installed': result={'apps':[]}
    elif method=='turn/start':
        if SCENARIO=='auth_expired' or (SCENARIO=='auth_recovery' and not os.path.exists(recovered_path)):
            send({'id':ident,'error':{'code':401,'message':'expired authentication; run codex login and retry'}});continue
        if SCENARIO=='rate_limit':
            send({'id':ident,'error':{'code':429,'message':'rate limit; retry later'}});continue
        send({'id':ident,'result':{'turn':{'id':'turn'}}})
        if SCENARIO in ['cancel','disconnect']:
            event('item/agentMessage/delta',{'threadId':'thread','delta':'partial output'})
            if SCENARIO=='disconnect':sys.exit(0)
            continue
        text='{"ok":true}'
        if SCENARIO=='malformed':text='not JSON'
        if SCENARIO=='two_repairs':
            try: count=int(open(counter_path).read())
            except FileNotFoundError: count=0
            with open(counter_path,'w') as f:f.write(str(count+1))
            text='{"ok":"wrong type"}' if count<2 else '{"ok":true}'
        item={'method':'item/completed','params':{'threadId':'thread','item':{'type':'agentMessage','phase':'final_answer','text':text}}}
        if SCENARIO=='chunked':
            serialized=json.dumps(item)+'\n';pivot=len(serialized)//2
            sys.stdout.write(serialized[:pivot]);sys.stdout.flush();time.sleep(.45)
            sys.stdout.write(serialized[pivot:]);sys.stdout.flush()
        else:send(item)
        completed();continue
    elif method=='turn/interrupt':
        send({'id':ident,'result':{}});completed('interrupted');continue
    send({'id':ident,'result':result})
