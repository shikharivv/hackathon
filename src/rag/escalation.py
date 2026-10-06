"""Confidence-based human hand-off; tickets are drafts, never sent automatically."""
import os
from datetime import datetime, timezone
from uuid import uuid4


def build_escalation(question, confidence):
    if confidence >= 0.60:
        return None
    contact = os.getenv('HR_CONTACT_EMAIL', 'hr-support@meridian.example')
    channel = os.getenv('HR_CONTACT_CHANNEL', 'HRIS > Support > HR Help')
    result = {'contact': contact, 'channel': channel,
              'demo_contact': contact.endswith('.example'),
              'action': 'ticket' if confidence < 0.45 else 'contact_hr'}
    if confidence < 0.45:
        ticket_id = 'HR-' + uuid4().hex[:10].upper()
        result['ticket'] = {
            'id': ticket_id, 'status': 'Draft — not sent to HR',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'question': question, 'confidence': confidence,
            'body': (f'Ticket: {ticket_id}\nStatus: Draft — not sent to HR\n'
                     f'Employee question: {question}\nPolicy match: {confidence:.1%}\n'
                     'Reason: Policy match below 45%; human review requested.\n'
                     'Please review this question and advise on the applicable policy.')}
    return result
