from pathlib import Path
from fastapi.testclient import TestClient
from src.rag.runtime import PolicyRuntime, ROOT
from src.api.main import create_app

def test_section_metadata_and_known_question(tmp_path):
    runtime = PolicyRuntime(log_path=tmp_path/'audit.sqlite3', generate=lambda q,s: 'See '+s[0]['policy_number'])
    result = runtime.query('annual leave full time employees entitlement', confidence_threshold=0.1)
    assert result['citations']
    assert result['citations'][0]['policy_number'] == 'POL-001'
    assert '3.1' in result['citations'][0]['section']
    assert result['citations'][0]['version'] == '3.2'
    assert (tmp_path/'audit.sqlite3').exists()

def test_unknown_no_generation(tmp_path):
    def fail(q,s):
        raise AssertionError('Unknown question must not call model')
    runtime = PolicyRuntime(log_path=tmp_path/'audit.sqlite3', generate=fail)
    result = runtime.query('quantum asteroid cryptocurrency')
    assert result['guardrail_flags'] == ['no_sources_found']

def test_invalid_citation_and_api_failure(tmp_path):
    runtime = PolicyRuntime(log_path=tmp_path/'audit.sqlite3', generate=lambda q,s:'According to POL-999, you get 900 days')
    result=runtime.query('annual leave full time entitlement', confidence_threshold=0.1)
    assert result['guardrail_flags'] == ['citation_validation_failed']
    def unavailable(q,s):
        raise RuntimeError('secret must never leak')
    runtime.generate=unavailable
    result=runtime.query('annual leave full time entitlement', confidence_threshold=0.1)
    assert result['guardrail_flags'] == ['llm_error']
    assert 'secret' not in result['answer']

def test_api_pipeline_and_index(monkeypatch,tmp_path):
    monkeypatch.setenv('AUDIT_DB_PATH',str(tmp_path/'audit.sqlite3'))
    monkeypatch.setenv('NVIDIA_API_KEY','')
    with TestClient(create_app()) as client:
        assert client.get('/api/v1/health').json()['documents_indexed']==6
        response=client.post('/api/v1/query',json={'question':'annual leave full time entitlement','confidence_threshold':0.1})
        assert response.status_code==200
        assert response.json()['citations']
        assert client.post('/api/v1/index').json()['documents_loaded']==6


def test_streamlit_frontend(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('AUDIT_DB_PATH', str(tmp_path/'ui.sqlite3'))
    monkeypatch.setenv('NVIDIA_API_KEY','')
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'HR Policy Assistant'
    app.chat_input[0].set_value('What is annual leave entitlement?').run(timeout=30)
    assert not app.exception
    assert len(app.chat_message)==2


def test_nvidia_request_configuration(monkeypatch,tmp_path):
    import io, json
    captured={}
    def response(request,timeout):
        captured['url']=request.full_url
        captured['payload']=json.loads(request.data)
        captured['authorization']=request.get_header('Authorization')
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':'See POL-001 section 3.1'}}]}).encode())
    monkeypatch.setenv('NVIDIA_API_KEY','test-only-placeholder')
    monkeypatch.setenv('NVIDIA_MODEL','test/model')
    monkeypatch.setattr('src.rag.runtime.urlopen',response)
    runtime=PolicyRuntime(log_path=tmp_path/'audit.sqlite3')
    result=runtime.query('annual leave full time entitlement',confidence_threshold=0.1)
    assert not result['fallback_triggered']
    assert captured['url']=='https://integrate.api.nvidia.com/v1/chat/completions'
    assert captured['payload']['model']=='test/model'
    assert captured['authorization']=='Bearer test-only-placeholder'
    assert '3.1 Full-Time Employees' in captured['payload']['messages'][1]['content']


def test_unicode_policy_identifier(tmp_path):
    runtime=PolicyRuntime(log_path=tmp_path/'audit.sqlite3',generate=lambda q,s:'See POL\u2011001 section 3.1')
    result=runtime.query('annual leave full time entitlement',confidence_threshold=0.1)
    assert not result['fallback_triggered']
