import pytest
from src.rag.escalation import build_escalation, create_ticket
from src.rag.runtime import PolicyRuntime


@pytest.mark.parametrize('score,action',[(0.0,'offer_ticket'),(0.4499,'offer_ticket'),(0.45,'contact_hr'),(0.5999,'contact_hr'),(0.60,None),(1.0,None)])
def test_threshold_boundaries(score,action):
    result=build_escalation('My policy question',score)
    if action is None:
        assert result is None
    else:
        assert result['action']==action
        assert 'ticket' not in result


def test_real_contact_configuration(monkeypatch):
    monkeypatch.setenv('HR_CONTACT_EMAIL','hr@demo.org')
    result=build_escalation('Question',0.2)
    assert result['contact']=='hr@demo.org'
    assert not result['demo_contact']
    assert create_ticket('Question',0.2)['status'].startswith('Draft')


def test_runtime_offers_ticket_for_unknown_question(tmp_path):
    runtime=PolicyRuntime(log_path=tmp_path/'audit.sqlite3')
    result=runtime.query('zzzzqqqqxxx')
    assert result['escalation']['action']=='offer_ticket'
    assert 'ticket' not in result['escalation']


def test_ui_ticket_survives_rerun(monkeypatch,tmp_path):
    from tests.test_workspace_navigation import open_app,click
    app=open_app(monkeypatch,tmp_path)
    app.chat_input[0].set_value('zzzzqqqqxxx').run(timeout=30)
    assert not app.exception
    assert 'ticket' not in app.session_state['messages'][-1]['extra']['escalation']
    next(b for b in app.button if b.label=='Create HR ticket').click().run(timeout=30)
    assert not app.exception
    ticket_id=app.session_state['messages'][-1]['extra']['escalation']['ticket']['id']
    assert any('HR ticket created' in s.value for s in app.subheader)
    click(app,'nav_contacts')
    click(app,'nav_chat')
    assert app.session_state['messages'][-1]['extra']['escalation']['ticket']['id']==ticket_id


def test_retained_18_percent_answer_gets_ticket(monkeypatch,tmp_path):
    from tests.test_workspace_navigation import open_app
    app=open_app(monkeypatch,tmp_path)
    app.session_state['messages']=[
        {'role':'user','content':'How many leave days per year?'},
        {'role':'assistant','content':'Please contact HR.',
         'extra':{'confidence':0.18,'citations':[],'answer_mode':'generated'}}]
    app.run(timeout=30)
    assert not app.exception
    assert 'ticket' not in app.session_state['messages'][-1]['extra']['escalation']
    next(b for b in app.button if b.label=='Create HR ticket').click().run(timeout=30)
    ticket=app.session_state['messages'][-1]['extra']['escalation']['ticket']
    assert ticket['question']=='How many leave days per year?'
    assert any('HR ticket created' in s.value for s in app.subheader)
    app.run(timeout=30)
    assert app.session_state['messages'][-1]['extra']['escalation']['ticket']['id']==ticket['id']
