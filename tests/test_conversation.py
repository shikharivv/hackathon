import json
import pytest
from src.rag.runtime import PolicyRuntime


@pytest.mark.parametrize('question',['hello','hi bro','thanks','how are you','who are you','what can you do'])
def test_small_talk_has_no_policy_controls(tmp_path,question):
    runtime=PolicyRuntime(generate=lambda *_:pytest.fail('Small talk should be immediate'),log_path=tmp_path/'log.sqlite3')
    result=runtime.query(question)
    assert result['answer']
    assert not result['show_confidence']
    assert result['escalation'] is None
    assert not result['citations']


def test_general_question_uses_general_model_path(tmp_path):
    calls=[]
    def generate(question,sources):
        calls.append(sources)
        return 'Paris is the capital of France.'
    result=PolicyRuntime(generate=generate,log_path=tmp_path/'log.sqlite3').query('What is the capital of France?')
    assert calls==[[]]
    assert result['answer']=='Paris is the capital of France.'
    assert result['escalation'] is None


def test_greeting_plus_policy_question_still_retrieves(tmp_path):
    result=PolicyRuntime(generate=lambda *_:'Apply through HRIS. [POL-001 §4.2]',log_path=tmp_path/'log.sqlite3').query('Hello, how can I apply for leave?')
    assert result['show_confidence']
    assert result['citations']


def test_reasoning_disabled_in_request(monkeypatch,tmp_path):
    monkeypatch.setenv('NVIDIA_API_KEY','test-only')
    def urlopen(request,timeout):
        payload=json.loads(request.data)
        assert payload['chat_template_kwargs']['enable_thinking'] is False
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b'{"choices":[{"message":{"content":"<think>hidden</think>Paris."}}]}'
        return Response()
    monkeypatch.setattr('src.rag.runtime.urlopen',urlopen)
    result=PolicyRuntime(log_path=tmp_path/'log.sqlite3').query('What is the capital of France?')
    assert result['answer']=='Paris.'


def test_greeting_ui_hides_score_and_ticket(monkeypatch,tmp_path):
    from tests.test_workspace_navigation import open_app
    app=open_app(monkeypatch,tmp_path)
    app.chat_input[0].set_value('hello').run(timeout=30)
    assert not app.exception
    assert not any('Confidence in policy match' in m.value for m in app.markdown)
    assert not any(b.label=='Create HR ticket' for b in app.button)
