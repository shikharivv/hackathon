import pytest
from src.rag.runtime import PolicyRuntime

@pytest.fixture
def runtime(tmp_path):
    return PolicyRuntime(log_path=tmp_path/'audit.sqlite3',generate=lambda q,s:('See '+s[0]['policy_number']) if s else ('Which type of leave do you need? Contact the HR Support Desk for help.'))

@pytest.mark.parametrize('question',[
    'how can i apply for leave','how do i apply for annual leave',
    'i want to take a day off','how do i request vacation',
    'where do i submit a leave request',
])
def test_leave_procedure_paraphrases(runtime,question):
    result=runtime.query(question)
    assert result['citations'][0]['section_id']=='POL-001#4.2'
    assert not result['fallback_triggered']


def test_every_faq_points_to_documented_sections(runtime):
    assert len(runtime.faqs)==30
    for faq in runtime.faqs:
        sources,confidence,_=runtime._retrieve(faq['questions'][0],5,0.15)
        assert [s['section_id'] for s in sources]==faq['sections'],faq['id']
        assert confidence>0.9


def test_vague_question_asks_clarification(runtime):
    result=runtime.query('leave')
    assert result['answer_mode']=='clarification'
    assert 'Which type of leave' in result['answer']
    assert result['suggested_questions']


def test_service_failure_keeps_helpful_policy_guidance(runtime):
    def fail(q,s):
        raise RuntimeError('private transport details')
    runtime.generate=fail
    result=runtime.query('how can i apply for leave')
    assert result['answer_mode']=='source_excerpt'
    assert 'HRIS' in result['answer']
    assert 'private transport' not in result['answer']


def test_unknown_does_not_bypass_threshold(runtime):
    result=runtime.query('quantum asteroid cryptocurrency')
    assert not result['citations']
    assert result['answer_mode']=='clarification'
    assert 'HR Support Desk' in result['answer']
