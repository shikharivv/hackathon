from __future__ import annotations

import os
from urllib.parse import urlencode
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from loguru import logger

from src.config.settings import get_settings
from src.rag.escalation import build_escalation, create_ticket

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

_EXAMPLE_QUESTIONS: list[str] = [
    "How can I apply for leave?",
    "Where do I find my payslip?",
    "Can I work remotely from another country?",
    "How do I swap shifts?",
    "How do I apply for sick leave?",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _api_available() -> bool:
    """Return *True* if the FastAPI backend is reachable."""
    try:
        resp = httpx.get(f"{_API_BASE}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _query_api(question: str, top_k: int) -> dict[str, Any]:
    """Send a query to the FastAPI backend and return the parsed response."""
    resp = httpx.post(
        f"{_API_BASE}/query",
        json={"question": question, "top_k": top_k, "include_sources": True, "confidence_threshold": st.session_state.confidence_threshold},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _query_direct(question: str, top_k: int) -> dict[str, Any]:
    """Fall back to invoking the RAG chain directly (no API server)."""
    return _runtime().query(question, top_k, st.session_state.confidence_threshold)


@st.cache_resource
def _runtime():
    from src.rag.runtime import PolicyRuntime
    return PolicyRuntime()


def _confidence_badge(confidence: float) -> str:
    label = 'Strong policy match' if confidence >= 0.65 else ('Policy match found' if confidence >= 0.15 else 'Needs clarification')
    return f"**Confidence in policy match: {confidence:.0%}** · {label}"


def _list_policies_on_disk() -> list[str]:
    """Return filenames of policy Markdown files found in docs/policies/."""
    settings = get_settings()
    policies_dir = Path(settings.vectorstore.path).parent.parent / "docs" / "policies"
    if not policies_dir.exists():
        return []
    return sorted(p.name for p in policies_dir.glob("*.md"))


def _reindex() -> str:
    """Trigger document re-indexing via the API."""
    if os.getenv('APP_MODE', 'direct') == 'direct':
        documents, sections = _runtime().reindex()
        return f'Indexed {documents} policies ({sections} sections).'
    try:
        resp = httpx.post(f"{_API_BASE}/index", timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Indexed **{data['documents_loaded']}** documents "
            f"(**{data['chunks_created']}** chunks) into "
            f"`{data['collection_name']}`."
        )
    except Exception as exc:
        return f"Re-indexing failed: {exc}"


# ------------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------------

def _save_conversation():
    if not st.session_state.messages:
        return
    chat_id=st.session_state.active_chat
    first=next((m['content'] for m in st.session_state.messages if m['role']=='user'),'Conversation')
    st.session_state.conversations[chat_id]={'title':first[:48], 'messages':deepcopy(st.session_state.messages)}


def _new_conversation():
    _save_conversation()
    st.session_state.chat_counter+=1
    st.session_state.active_chat=f'chat_{st.session_state.chat_counter}'
    st.session_state.messages=[]
    st.session_state.view='chat'


def _open_conversation(chat_id):
    _save_conversation()
    st.session_state.active_chat=chat_id
    st.session_state.messages=deepcopy(st.session_state.conversations[chat_id]['messages'])
    st.session_state.view='chat'


def _ensure_escalations():
    """Upgrade retained responses or older API results once, keeping ticket IDs stable."""
    question=''
    changed=False
    for message in st.session_state.messages:
        if message['role']=='user':
            question=message['content']
        elif message.get('extra') is not None:
            data=message['extra']
            if 'escalation' not in data:
                data['escalation']=build_escalation(question,float(data.get('confidence',0)))
                changed=True
    if changed:
        _save_conversation()


def main() -> None:
    st.set_page_config(page_title='HR Policy Assistant',page_icon='💬',layout='wide')
    st.markdown("""<style>
    .stApp {background:#08090b;}
    [data-testid="stHeader"] {background:transparent;}
    .block-container {max-width:1080px;padding:2.5rem 2.2rem 6rem;}
    h1,h2,h3 {color:#f2f3f5;letter-spacing:-1px;}
    h1 {font-size:25px!important;font-weight:650!important;}
    [data-testid="stSidebar"] {background:#101215;border-right:1px solid #25282c;}
    [data-testid="stSidebar"] .stButton button {width:100%;border-radius:9px;justify-content:flex-start;font-size:14px;}
    .stButton button {border:1px solid #30343a;border-radius:12px;background:#13161a;color:#e6e8eb;min-height:43px;transition:background .15s,border-color .15s;}
    .stButton button:hover {background:#1b2524;border-color:#5a9485;color:#d9fff1;}
    .stButton button[kind="primary"] {background:#184538;border-color:#377b63;color:#e5fff3;}
    [data-testid="stChatMessage"] {border:1px solid #2b3037;border-radius:16px;background:#101317;padding:1.4rem;margin-bottom:1.2rem;}
    [data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li {line-height:1.8;font-size:15px;}
    [data-testid="stChatInput"] {border:1px solid #414950;border-radius:16px;background:#14181d;}
    [data-testid="stBottom"]>div {background:#08090b;}
    [data-testid="stExpander"] {border-color:#2a3036;border-radius:12px;background:#101317;}
    [data-testid="stCaptionContainer"] {color:#8f99a6;}
    .brand {display:flex;align-items:center;gap:12px;margin:8px 0 24px;}
    .brand-mark {background:#dcece6;color:#10251d;border-radius:9px;padding:10px 8px;font-weight:850;font-size:15px;}
    .brand-name {font-size:16px;font-weight:700;color:#eceff2;}
    .brand-sub {font-size:11px;color:#87928e;margin-top:3px;letter-spacing:1px;text-transform:uppercase;}
    .eyebrow {color:#8caaa0;font-size:11px;font-weight:650;letter-spacing:2px;margin-bottom:8px;}
    .hero {position:relative;overflow:hidden;padding:58px 0 30px;}
    .hero::after {content:'';position:absolute;right:0;top:4px;width:270px;height:270px;background:radial-gradient(circle,rgba(167,206,189,.11) 0%,rgba(90,137,116,.06) 35%,transparent 70%);pointer-events:none;}
    .hero-title {position:relative;z-index:1;font-size:clamp(40px,5.2vw,68px);line-height:1.08;letter-spacing:-3px;font-weight:600;color:#f0f2f4;margin-bottom:24px;}
    .hero-title span {color:#90999f;}
    .hero-copy {position:relative;z-index:1;max-width:530px;color:#a7afb9;font-size:16px;line-height:1.8;margin-bottom:18px;}
    .trust-row {display:flex;flex-wrap:wrap;gap:22px;color:#9ca9a4;font-size:12px;margin:14px 0 32px;}
    .trust-row span::before {content:'·';color:#72c8a5;font-size:20px;margin-right:8px;}
    .confidence-card {display:flex;align-items:center;justify-content:space-between;border-radius:12px;padding:13px 18px;margin-bottom:18px;border:1px solid;}
    .confidence-card .score {font-size:30px;font-weight:750;letter-spacing:-1px;}
    .confidence-card .label {font-size:13px;font-weight:650;}
    .confidence-card .detail {font-size:11px;margin-top:4px;opacity:.8;}
    .strong {background:#102a22;border-color:#346550;color:#a4e8c8;}
    .moderate {background:#292316;border-color:#63502b;color:#e6c989;}
    .unclear {background:#1b2029;border-color:#394455;color:#bdc9da;}
    @media (max-width:700px) {.block-container {padding:1.5rem 1rem 6rem;} .hero {padding-top:30px;} .hero-title {letter-spacing:-1.7px;} .hero::after {width:170px;height:170px;} .trust-row {gap:12px;}}
    @media (prefers-reduced-motion:reduce) {.stButton button {transition:none;}}
    </style>""",unsafe_allow_html=True)
    for key,value in {'messages':[],'top_k':5,'confidence_threshold':float(os.getenv('CONFIDENCE_THRESHOLD','0.15')),'view':'chat','conversations':{},'chat_counter':0,'active_chat':'chat_0'}.items():
        st.session_state.setdefault(key,value)
    _ensure_escalations()
    runtime=_runtime()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">HR</div><div><div class="brand-name">Policy Assistant</div><div class="brand-sub">Employee workspace</div></div></div>',unsafe_allow_html=True)
        if st.button('＋ New conversation',key='new_conversation',use_container_width=True):
            _new_conversation()
            st.rerun()
        st.divider()
        for key,label,view in [('nav_chat','Chat assistant','chat'),('nav_policies','Policy library','policies'),('nav_contacts','HR contacts','contacts')]:
            if st.button(label,key=key,type='primary' if st.session_state.view==view else 'secondary',use_container_width=True):
                st.session_state.view=view
                st.rerun()
        st.divider()
        st.caption('RECENT CONVERSATIONS')
        chats=list(st.session_state.conversations.items())[-8:][::-1]
        if not chats:
            st.caption('Your conversations will appear here.')
        for chat_id,chat in chats:
            if st.button(chat['title'],key='recent_'+chat_id,use_container_width=True):
                _open_conversation(chat_id)
                st.rerun()
        st.caption('Chats are kept while this page stays open.')
        st.divider()
        st.caption('Demo workspace · Sample company policies')
    st.markdown('<div class="eyebrow">EMPLOYEE SUPPORT</div>',unsafe_allow_html=True)
    st.title('HR Policy Assistant')
    if st.session_state.view=='policies':
        _policy_library(runtime)
        return
    if st.session_state.view=='contacts':
        _hr_contacts(runtime)
        return
    welcome=not st.session_state.messages
    prompt=None
    if welcome:
        st.markdown('<div class="hero"><div class="hero-title">Your HR questions.<br><span>Answered.</span></div><div class="hero-copy">Less searching. More clarity.<br>Find the policy, understand your options, and know what to do next.</div></div><div class="trust-row"><span>Policy-backed answers</span><span>Clear next steps</span><span>HR contact guidance</span></div>',unsafe_allow_html=True)
        with st.container(border=True):
            prompt=st.chat_input('Ask a question about work…',key='question_input')
        st.caption('Or start with a common question')
        for column,label,question in zip(st.columns(3),['Leave & time off','Pay & expenses','Shifts & attendance'],['How can I apply for leave?','Where do I find my payslip?','How do I swap shifts?']):
            if column.button(label,key='quick_'+label,use_container_width=True):
                with st.spinner('Finding the right policy…'):
                    _handle_question(question)
                st.rerun()
    else:
        st.caption('Clear guidance, with the policy behind every answer.')
        for index,msg in enumerate(st.session_state.messages):
            with st.chat_message(msg['role']):
                if msg['role']=='assistant' and msg.get('extra'):
                    _render_confidence(msg['extra'])
                st.markdown(msg['content'])
                if msg['role']=='assistant' and msg.get('extra'):
                    _render_extra(msg['extra'],f'{st.session_state.active_chat}_{index}')
        prompt=st.chat_input('Ask a follow-up or a new question…',key='question_input')
    if prompt:
        with st.spinner('Finding the right policy…'):
            _handle_question(prompt)
        st.rerun()


def _policy_library(runtime):
    st.header('Find your policy')
    st.caption('Search the handbook by title, topic, or a phrase. These are fictional demo policies.')
    query=st.text_input('Search policies',placeholder='Try sick leave, overtime, or reimbursement',key='policy_search').strip().lower()
    matches=[section for section in runtime.sections if not query or query in (section['policy_title']+' '+section['section']+' '+section['chunk_text']).lower()]
    titles=sorted({section['policy_title'] for section in matches})
    if not titles:
        st.info('No matching policy yet. Try another keyword, or use HR contacts to find the right team.')
        return
    selected=st.selectbox('Choose a policy',titles,key='policy_choice')
    sections=[section for section in matches if section['policy_title']==selected]
    st.caption(f"{sections[0]['policy_number']} · Version {sections[0]['version']} · {len(sections)} matching sections")
    for section in sections:
        with st.expander(section['section']):
            st.markdown(section['chunk_text'])


def _hr_contacts(runtime):
    st.header('Find the right people')
    st.caption('Use these documented channels when you need someone to review your situation.')
    for section in runtime.sections:
        if section['policy_number']=='POL-012' and section['section_id']!='POL-012#5':
            with st.container(border=True):
                st.subheader(section['section'].split('. ',1)[-1])
                st.markdown(section['chunk_text'])
                st.caption(f"{section['policy_number']} · {section['section']} · Version {section['version']}")


def _render_confidence(data):
    if data.get('show_confidence') is False or data.get('answer_mode')=='general':
        return
    score=min(1.0,max(0.0,float(data.get('confidence',0))))
    if score<0.45:
        theme,label='unclear','HR review recommended · Ticket available'
    elif data.get('answer_mode')=='clarification':
        theme,label='unclear','A little more detail will help'
    elif score>=0.60:
        theme,label='strong','Strong policy match'
    else:
        theme,label=('unclear','HR ticket drafted') if score<0.45 else ('moderate','HR review recommended')
    st.markdown(f'<div class="confidence-card {theme}"><div><div class="label">Confidence in policy match</div><div class="detail">{label}</div></div><div class="score">{score:.0%}</div></div>',unsafe_allow_html=True)
    st.caption('Match strength, not a guarantee of answer accuracy.')


# ------------------------------------------------------------------
# Chat logic
# ------------------------------------------------------------------

def _handle_question(question: str) -> None:
    """Process *question*: call backend, store messages, and display."""

    st.session_state.messages.append({"role": "user", "content": question})

    use_api = os.getenv("APP_MODE", "direct") == "api" and _api_available()
    try:
        if use_api:
            data = _query_api(question, st.session_state.top_k)
        else:
            data = _query_direct(question, st.session_state.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        data = {
            "answer": f"An error occurred: {exc}",
            "citations": [],
            "confidence": 0.0,
            "sources_used": 0,
            "fallback_triggered": True,
            "guardrail_flags": ["error"],
            "latency_ms": 0.0,
        }

    if 'escalation' not in data:
        data['escalation']=build_escalation(question,float(data.get('confidence',0)))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "extra": data,
        }
    )

    _save_conversation()


def _render_extra(data: dict[str, Any], message_key: str = 'answer') -> None:
    """Render citations, confidence badge, and fallback notice."""

    escalation=data.get('escalation')
    if escalation:
        st.warning('The policy match is below 60%. Please contact HR to confirm guidance for your situation.')
        st.markdown(f"**HR contact:** {escalation['contact']}  \n**Support channel:** {escalation['channel']}")
        if escalation.get('demo_contact'):
            st.caption('Demo contact only: this address is not monitored. Use your company HR channel.')
        ticket=escalation.get('ticket')
        if escalation.get('action')=='offer_ticket' and not ticket:
            if st.button('Create HR ticket',key='create_ticket_'+message_key):
                escalation['ticket']=create_ticket(escalation['question'],float(data.get('confidence',0)))
                _save_conversation()
                st.rerun()
        if ticket:
            with st.container(border=True):
                st.subheader('HR ticket created')
                st.caption(f"{ticket['id']} · {ticket['status']}")
                st.write('You created this ticket for HR review. You can download it or forward it to HR.')
                st.text(ticket['body'])
                st.download_button('Download ticket',ticket['body'],file_name=ticket['id']+'.txt',mime='text/plain',key='ticket_'+message_key)
                if not escalation.get('demo_contact'):
                    params=urlencode({'subject':ticket['id']+' — HR policy review','body':ticket['body']})
                    st.link_button('Forward to HR in your email app','mailto:'+escalation['contact']+'?'+params)
                    st.caption('Opens an email draft; you review and send it yourself.')
    if data.get('answer_mode') == 'source_excerpt':
        st.caption('Showing the documented guidance directly so you can still get help.')
    for index,suggestion in enumerate(data.get('suggested_questions',[])):
        if st.button(suggestion,key=f'follow_{message_key}_{index}'):
            _handle_question(suggestion)
            st.rerun()
    citations = data.get('citations', [])
    if citations:
        with st.expander(f"View policy sources ({len(citations)})"):
            for cit in citations:
                st.markdown('**'+cit.get('policy_title','Policy')+'**')
                st.caption(cit.get('policy_number','')+' · '+cit.get('section','')+' · Version '+cit.get('version',''))
                if cit.get('chunk_text'):
                    st.markdown(cit['chunk_text'])
                st.divider()

    # Latency
    latency = data.get("latency_ms", 0.0)
    if latency:
        suffix=' · Based on documented policies' if data.get('answer_mode')!='general' else ''
        st.caption(f"Answered in {latency/1000:.1f}s{suffix}")


# ------------------------------------------------------------------
# Script entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
