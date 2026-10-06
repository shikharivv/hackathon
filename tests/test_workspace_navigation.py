from streamlit.testing.v1 import AppTest
from src.rag.runtime import ROOT


def open_app(monkeypatch,tmp_path):
    monkeypatch.setenv('NVIDIA_API_KEY','')
    monkeypatch.setenv('AUDIT_DB_PATH',str(tmp_path/'audit.sqlite3'))
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    return app


def click(app,key):
    next(button for button in app.button if button.key==key).click().run(timeout=30)
    assert not app.exception


def test_recent_chats_restore_independent_conversations(monkeypatch,tmp_path):
    app=open_app(monkeypatch,tmp_path)
    assert not app.slider
    app.chat_input[0].set_value('how can i apply for leave').run(timeout=30)
    click(app,'new_conversation')
    assert not app.chat_message
    app.chat_input[0].set_value('where do i find my payslip').run(timeout=30)
    click(app,'recent_chat_0')
    assert app.session_state['messages'][0]['content']=='how can i apply for leave'
    assert len(app.chat_message)==2
    click(app,'recent_chat_1')
    assert app.session_state['messages'][0]['content']=='where do i find my payslip'


def test_policy_search_and_hr_contact_navigation(monkeypatch,tmp_path):
    app=open_app(monkeypatch,tmp_path)
    click(app,'nav_policies')
    app.text_input[0].set_value('sick').run(timeout=30)
    assert 'Sick and Emergency Leave Requests' in app.selectbox[0].options
    app.selectbox[0].select('Sick and Emergency Leave Requests').run(timeout=30)
    assert any('Applying for Sick Leave' in element.label for element in app.expander)
    app.text_input[0].set_value('zzzzzzzzzzzzz').run(timeout=30)
    assert any('No matching policy' in element.value for element in app.info)
    click(app,'nav_contacts')
    assert any('hr-support@meridian.example' in element.value for element in app.markdown)
    click(app,'nav_chat')
    assert app.chat_input


def test_clarification_suggestion_starts_supported_followup(monkeypatch,tmp_path):
    app=open_app(monkeypatch,tmp_path)
    app.chat_input[0].set_value('leave').run(timeout=30)
    next(button for button in app.button if button.label=='How can I apply for leave?').click().run(timeout=30)
    assert not app.exception
    assert len(app.chat_message)==4
    assert app.session_state['messages'][-1]['extra']['citations'][0]['section_id']=='POL-001#4.2'
