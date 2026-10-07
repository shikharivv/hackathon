"""Small talk and general questions bypass policy scoring and escalation."""
import re

GENERAL_PROMPT = ('You are a friendly assistant in an HR support app. Answer general questions '
                  'briefly and clearly. Do not invent company policies or personal employee data. '
                  'For company-specific guidance, ask the user to ask a policy question. '
                  'Give only the final answer, without reasoning, thinking tags, citations, or scores.')


def conversation_route(question):
    text=re.sub(r'[^a-z0-9\s]', ' ', question.lower())
    text=' '.join(text.split())
    # Greetings attached to a policy question must still use policy retrieval.
    if re.search(r'\b(leave|time off|days off|vacation|holiday|sick|pay|payroll|salary|payslip|expense|reimburse\w*|shift\w*|roster|attendance|benefit\w*|remote|overtime|policy|policies|employee|employer|company|manager|hris|hr|workplace|onboard\w*|resign\w*|uniform|discount|badge|conduct|harass\w*|entitlement)\b',text):
        return None
    if re.fullmatch(r'(hello|hi|hey|heyy|hello there|hi there|good morning|good afternoon|good evening)( bro| assistant)?',text):
        return 'Hello! How can I help you today? You can ask a general question or get help with an HR policy.'
    if text in {'thanks','thank you','thank you bro','thanks bro','ok','okay','bye','goodbye'}:
        return "You're welcome! I'm here whenever you need help."
    if text in {'how are you','how are you doing','whats up','what s up'}:
        return "I'm ready to help! What would you like to know?"
    if text in {'help','what can you do','who are you'}:
        return 'I can answer general questions and help you find HR policies, forms, and support contacts. What do you need help with?'
    if re.match(r'^(what|who|where|when|why|how|is|are|can|tell me|explain|define)\b',text):
        return ''  # Generate a general answer without policy citations.
    return None
