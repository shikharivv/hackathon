import io
import json
import pytest
from src.rag.runtime import PolicyRuntime


@pytest.mark.parametrize('question',['hello','hi bro','thanks','how are you','who are you','what can you do'])
def test_small_talk_is_model_decision(tmp_path,question):
    calls=[]
    def classify(q):
        calls.append(q)
        return {'route':'general','answer':'A unique AI reply to '+q}
    result=PolicyRuntime(classify=classify,log_path=tmp_path/'log.sqlite3').query(question)
    assert calls==[question]
    assert result['answer']=='A unique AI reply to '+question
    assert not result['show_confidence']
    assert result['escalation'] is None


def test_no_keyword_gate_for_work_question(tmp_path):
    result=PolicyRuntime(classify=lambda q:{'route':'work','search_query':'annual leave request'},
                         generate=lambda q,s:'Apply through HRIS. [POL-001]',
                         log_path=tmp_path/'log.sqlite3').query('Need tomorrow away from the shop, what do I do?')
    assert result['citations']
    assert result['show_confidence']


def test_model_can_route_work_keyword_to_general(tmp_path):
    result=PolicyRuntime(classify=lambda q:{'route':'general','answer':'Work means effort toward a goal.'},
                         log_path=tmp_path/'log.sqlite3').query('What does the word work mean?')
    assert not result['show_confidence']
    assert result['escalation'] is None


def test_routing_and_reasoning_configuration(monkeypatch,tmp_path):
    monkeypatch.setenv('NVIDIA_API_KEY','test-only')
    def urlopen(request,timeout):
        payload=json.loads(request.data)
        assert payload['chat_template_kwargs']['enable_thinking'] is True
        assert payload['response_format']=={'type':'json_object'}
        content=json.dumps({'route':'general','answer':'Hello from the AI!','search_query':''})
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':content}}]}).encode())
    monkeypatch.setattr('src.rag.runtime.urlopen',urlopen)
    result=PolicyRuntime(log_path=tmp_path/'log.sqlite3').query('hello')
    assert result['answer']=='Hello from the AI!'


def test_irrelevant_policy_is_rejected_by_ai(monkeypatch,tmp_path):
    monkeypatch.setenv('NVIDIA_API_KEY','test-only')
    calls=[]
    def urlopen(request,timeout):
        payload=json.loads(request.data)
        calls.append(payload)
        assert payload['chat_template_kwargs']['enable_thinking'] is True
        if len(calls)==1:
            content=json.dumps({'route':'work','answer':'','search_query':'expense reimbursement'})
        elif len(calls)==2:
            content=json.dumps({'section_ids':[]})
        else:
            content='I cannot confirm pet adoption reimbursement from these policies. Contact HR; you can create a ticket below.'
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':content}}]}).encode())
    monkeypatch.setattr('src.rag.runtime.urlopen',urlopen)
    result=PolicyRuntime(log_path=tmp_path/'log.sqlite3').query('Can the shop cover pet adoption?')
    assert len(calls)==3
    assert not result['citations']
    assert result['confidence']==0
    assert result['escalation']['action']=='offer_ticket'
    assert 'ticket' not in result['escalation']
    assert 'pet adoption' in result['answer']


def test_missing_key_does_not_pretend_to_be_ai(monkeypatch,tmp_path):
    monkeypatch.setenv('NVIDIA_API_KEY','')
    result=PolicyRuntime(log_path=tmp_path/'log.sqlite3').query('hello')
    assert result['guardrail_flags']==['routing_unavailable']
    assert result['answer_mode']=='error'
    assert not result['show_confidence']
    assert result['escalation'] is None


def test_greeting_ui_hides_score_and_ticket(monkeypatch,tmp_path):
    from tests.test_workspace_navigation import open_app
    monkeypatch.setattr(PolicyRuntime,'_classify',lambda self,q:{'route':'general','answer':'Hey! What can I help with?'})
    app=open_app(monkeypatch,tmp_path)
    app.chat_input[0].set_value('hello').run(timeout=30)
    assert not app.exception
    assert not any('Confidence in policy match' in m.value for m in app.markdown)
    assert not any(b.label=='Create HR ticket' for b in app.button)
