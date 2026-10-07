"""CPU-only, section-based runtime for hosted demos; no GPU/model download."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from src.rag.answer_prompt import SYSTEM_PROMPT
from src.rag.escalation import build_escalation
from src.rag.conversation import ROUTING_PROMPT, GENERAL_PROMPT, RELEVANCE_PROMPT, MISSING_POLICY_PROMPT

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')

class PolicyRuntime:
    def __init__(self, policies_dir=None, generate=None, log_path=None, classify=None):
        self.policies_dir = Path(policies_dir or ROOT / 'docs/policies')
        self.generate = generate or self._generate
        self.classify = classify or self._classify
        self.log_path = Path(log_path or os.getenv('AUDIT_DB_PATH', str(ROOT / 'data/audit.sqlite3')))
        self.reindex()

    def reindex(self):
        self.sections = []
        files = sorted(self.policies_dir.glob('*.md'))
        for path in files:
            text = path.read_text(encoding='utf-8')
            title = re.search(r'^# (.+)$', text, re.M)
            number = re.search(r'Policy Number:\*\*\s*([^\n]+)', text)
            version = re.search(r'Version:\*\*\s*([^\n]+)', text)
            parent = ''
            for match in re.finditer(r'^(#{2,3}) (.+)\n([\s\S]*?)(?=^#{2,3} |\Z)', text, re.M):
                heading, body = match.group(2).strip(), match.group(3).strip()
                if match.group(1) == '##':
                    parent = heading
                if not body:
                    continue
                self.sections.append(dict(policy_number=number.group(1).strip() if number else path.stem,
                    policy_title=title.group(1) if title else path.stem,
                    section=heading if match.group(1) == '##' else parent+' / '+heading,
                    version=version.group(1).strip() if version else 'unknown',
                    section_id=(number.group(1).strip() if number else path.stem)+'#'+heading.split()[0].rstrip('.'),
                    source_file=path.name, chunk_text=body))
        if not self.sections:
            raise ValueError('No policy sections found')
        self.vectorizer = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS - {'full', 'part'}), ngram_range=(1,2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([(s['policy_title']+' '+s['section']+' ')*3+s['chunk_text'] for s in self.sections])
        self.heading_matrix = self.vectorizer.transform([s['section'] for s in self.sections])
        self.section_lookup = {section['section_id']: section for section in self.sections}
        faq_path = self.policies_dir.parent / 'faq.json'
        self.faqs = json.loads(faq_path.read_text(encoding='utf-8')) if faq_path.exists() else []
        self.faq_questions, self.faq_owners = [], []
        for index, faq in enumerate(self.faqs):
            for ref in faq['sections']:
                if ref not in self.section_lookup:
                    raise ValueError('FAQ references missing policy section: '+ref)
            for question in faq['questions']:
                self.faq_questions.append(question)
                self.faq_owners.append(index)
        if self.faq_questions:
            self.faq_vectorizer = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS - {'full', 'part'}), ngram_range=(1,2))
            self.faq_matrix = self.faq_vectorizer.fit_transform(self.faq_questions)
            self.faq_chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
            self.faq_char_matrix = self.faq_chars.fit_transform(self.faq_questions)
        self.documents_indexed = len(files)
        return len(files), len(self.sections)

    def _request(self, system, message, json_output=False):
        key = os.getenv('NVIDIA_API_KEY', '')
        if not key:
            raise ValueError('NVIDIA_API_KEY is not configured')
        payload = dict(model=os.getenv('NVIDIA_MODEL', 'nvidia/nemotron-3-super-120b-a12b'),
                       temperature=0.3, top_p=1, max_tokens=8192, stream=False,
                       messages=[dict(role='system',content=system),dict(role='user',content=message)],
                       chat_template_kwargs={'enable_thinking':True})
        if json_output:
            payload['response_format']={'type':'json_object'}
        request=Request('https://integrate.api.nvidia.com/v1/chat/completions',
                        data=json.dumps(payload).encode(),
                        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=90) as response:
                content=json.load(response)['choices'][0]['message']['content'] or ''
            content=re.sub(r'<think\b[^>]*>[\s\S]*?</think>','',content,flags=re.I).strip()
            if not content:
                raise RuntimeError('Empty model response')
            return json.loads(content) if json_output else content
        except HTTPError as exc:
            raise RuntimeError(f'NVIDIA returned HTTP {exc.code}; check key, model access, and quota') from None
        except (URLError,TimeoutError):
            raise RuntimeError('NVIDIA service is unavailable; try again later') from None

    def _classify(self, question):
        decision=self._request(ROUTING_PROMPT,question,json_output=True)
        if not isinstance(decision,dict) or decision.get('route') not in {'general','work'}:
            raise ValueError('Invalid AI routing decision')
        if decision['route']=='general' and (not isinstance(decision.get('answer'),str) or not decision['answer'].strip()):
            raise ValueError('Invalid general answer')
        return decision

    @staticmethod
    def _context(sources):
        return '\n\n'.join(f"[{s['section_id']} | {s['section']} | v{s['version']}]\n{s['chunk_text']}" for s in sources)

    def _generate(self, question, sources):
        return self._request(SYSTEM_PROMPT if sources else GENERAL_PROMPT,
                             f'POLICY EXCERPTS:\n{self._context(sources)}\n\nQUESTION:\n{question}' if sources else question)

    def query(self, question, top_k=5, confidence_threshold=None):
        start=time.perf_counter()
        try:
            decision=self.classify(question)
        except Exception:
            result=dict(answer='The AI service is unavailable, so I could not assess your question. Please try again shortly.',
                        citations=[],confidence=0.0,sources_used=0,fallback_triggered=True,
                        guardrail_flags=['routing_unavailable'],latency_ms=round((time.perf_counter()-start)*1000,2),
                        answer_mode='error',show_confidence=False,suggested_questions=[],escalation=None)
            self._log(question,result)
            return result
        if decision['route']=='general':
            answer=decision.get('answer','').strip()
            if not answer:
                answer=self.generate(question,[])
            result=dict(answer=answer,citations=[],confidence=0.0,sources_used=0,
                        fallback_triggered=False,guardrail_flags=[],latency_ms=round((time.perf_counter()-start)*1000,2),
                        answer_mode='general',show_confidence=False,suggested_questions=[],escalation=None)
            self._log(question,result)
            return result
        threshold=float(confidence_threshold if confidence_threshold is not None else os.getenv('CONFIDENCE_THRESHOLD','0.15'))
        search=decision.get('search_query') or question
        sources,confidence,suggestions=self._retrieve(search,top_k,threshold)
        flags=[]
        try:
            if sources and self.generate==self._generate:
                review=self._request(RELEVANCE_PROMPT,f'QUESTION: {question}\nCANDIDATES:\n{self._context(sources)}',json_output=True)
                refs=review.get('section_ids')
                if not isinstance(refs,list) or not all(isinstance(ref,str) and ref in {s['section_id'] for s in sources} for ref in refs):
                    raise ValueError('Invalid policy review')
                sources=[s for s in sources if s['section_id'] in refs]
            if not sources:
                confidence=0.0
                escalation=build_escalation(question,confidence)
                answer=self._request(MISSING_POLICY_PROMPT,
                                     json.dumps({'question':question,'contact':escalation['contact'],'channel':escalation['channel'],'demo_contact':escalation['demo_contact']})) if self.generate==self._generate else self.generate(question,[])
                flags=['no_sources_found']
                mode='clarification'
            else:
                answer=self.generate(question,sources)
                mentioned=set(re.findall(r'POL-\d+',answer.translate(str.maketrans({'\u2010':'-','\u2011':'-','\u2013':'-','\u2014':'-'}))))
                if not mentioned or not mentioned.issubset({s['policy_number'] for s in sources}):
                    answer=self._excerpt_answer(sources)
                    flags=['citation_validation_failed']
                mode='source_excerpt' if flags else 'generated'
        except Exception:
            flags=['llm_error']
            if sources:
                answer=self._excerpt_answer(sources)
                mode='source_excerpt'
            else:
                confidence=0.0
                answer='The AI service could not complete this response. Please try again, or use the HR contact below.'
                mode='error'
        result=dict(answer=answer,citations=sources,confidence=confidence,sources_used=len(sources),
                    fallback_triggered=bool(flags),guardrail_flags=flags,latency_ms=round((time.perf_counter()-start)*1000,2),
                    answer_mode=mode,suggested_questions=suggestions if not sources else [],
                    escalation=build_escalation(question,confidence),show_confidence=True)
        self._log(question,result)
        return result

    def _retrieve(self, question, top_k, threshold):
        query_vector = self.vectorizer.transform([question])
        scores = (0.75 * cosine_similarity(query_vector, self.matrix)[0]
                  + 0.25 * cosine_similarity(query_vector, self.heading_matrix)[0])
        confidence = float(max(scores))
        suggestions = []
        if self.faq_questions:
            word_scores = cosine_similarity(self.faq_vectorizer.transform([question]), self.faq_matrix)[0]
            char_scores = cosine_similarity(self.faq_chars.transform([question]), self.faq_char_matrix)[0]
            faq_scores = [0.0]*len(self.faqs)
            for index, score in enumerate(0.8*word_scores + 0.2*char_scores):
                owner = self.faq_owners[index]
                faq_scores[owner] = max(faq_scores[owner],float(score))
            ranked = sorted(range(len(self.faqs)),key=lambda i:faq_scores[i],reverse=True)
            best = faq_scores[ranked[0]]
            runner_up = faq_scores[ranked[1]] if len(ranked)>1 else 0.0
            suggestions = [self.faqs[i]['questions'][0] for i in ranked[:3] if faq_scores[i]>0.1]
            # A FAQ route requires a strong, distinct question match.
            if best>=max(0.42,threshold) and best-runner_up>=0.04:
                refs = self.faqs[ranked[0]]['sections']
                return ([{**self.section_lookup[ref], 'relevance_score':round(best,4)} for ref in refs],best,[])
            confidence = max(confidence,best)
        selected = [int(i) for i in scores.argsort()[::-1][:top_k] if scores[i]>=threshold]
        meaningful = self.vectorizer.build_analyzer()(question)
        if len([word for word in meaningful if ' ' not in word])<2:
            selected=[]
        sources=[{**self.sections[i],'relevance_score':round(float(scores[i]),4)} for i in selected]
        if not sources and not suggestions:
            suggestions=['How can I apply for leave?', 'Where do I find my payslip?', 'How do I swap shifts?']
        return sources,confidence,suggestions if not sources else []

    @staticmethod
    def _excerpt_answer(sources):
        excerpts='\n\n'.join(f"**{s['section']}**\n{s['chunk_text']}\n[{s['policy_number']} | {s['section']} | v{s['version']}]" for s in sources)
        return "Here is the guidance I found in the policy:\n\n"+excerpts

    def _log(self, question, result):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.log_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS queries (timestamp TEXT, question TEXT, answer TEXT, citations TEXT, confidence REAL, flags TEXT)')
            db.execute('INSERT INTO queries VALUES (?,?,?,?,?,?)', (time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),question,result['answer'],json.dumps(result['citations']),result['confidence'],json.dumps(result['guardrail_flags'])))
