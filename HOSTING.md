# NVIDIA setup and free hosting

Project maintained at https://github.com/shikharivv/hackathon.
README.md describes the current hackathon demo and serving architecture.
The MIT license notices are retained in LICENSE.

## What changed
The serving path now uses local scikit-learn TF-IDF retrieval plus 30 curated FAQ mappings over exact Markdown
sections and NVIDIA hosted inference, avoiding GPU and large model downloads.
The original LangChain/Chroma modules are retained as reference, but are not used
by the hosted runtime. The new evidence threshold defaults to 0.15 and must be
tuned; it is not an answer accuracy percentage. NVIDIA failures return a clean
fallback with source excerpts. With no key, the app shows source excerpts only.
All demo policies are visible. Below 60% policy match, the app recommends HR contact;
below 45% it automatically creates a downloadable ticket draft with a unique ID.
Tickets are not submitted to an HR system. A configured real email enables an email draft
for the employee to review and send. Citation checks validate document IDs, not every factual
claim or section ID. Local SQLite audit logs are ephemeral on free cloud hosting.

## Local verification
Use Python 3.12. Create a virtual environment and install requirements-hosted.txt.
Copy .env.example to .env and set NVIDIA_API_KEY. Choose an available model ID
from your NVIDIA Build account for NVIDIA_MODEL. Never commit .env.
Run: python -m streamlit run app.py
Optional API: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Use APP_MODE=api only when the API is running; direct mode is the hosting default.

## Streamlit Community Cloud
1. Push this modified project to your GitHub repository, preserving LICENSE.
2. Open https://share.streamlit.io and connect the repository.
3. Select app.py as the entrypoint and Python 3.12.
4. This repository's root requirements.txt contains the lightweight hosted dependencies.
5. In Advanced settings / Secrets enter server-side TOML:

```toml
NVIDIA_API_KEY = "your-secret-key"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
CONFIDENCE_THRESHOLD = "0.15"
HR_CONTACT_EMAIL = "your-hr-email@example.org"
HR_CONTACT_CHANNEL = "Your approved HR support channel"
```

6. Deploy and test a known policy question and an unsupported question.
No separate FastAPI deployment or NVIDIA GPU is required.

Streamlit Community Cloud is free with resource limits. Local files may disappear
on restart; use an external database (for example Supabase) for persistent audit
logs. This version does not yet include that database integration.
NVIDIA free endpoint access is for prototyping; check account availability and
terms before production use. API quotas and model availability can change.
Hackathon rules may impose additional originality or attribution requirements.

## Updated knowledge base and answer behavior

The demo includes 12 policy documents and 30 FAQ mappings in docs/faq.json.
Common question variants route to explicit section IDs; other questions use
section search. FAQ routing requires a distinct match of at least 0.42 and
respects a higher configured threshold. Vague topics ask for clarification.
The confidence display is a lexical question-to-policy match score, not a
calibrated probability of answer correctness. Scores near 100% can reflect
exactly matching a curated FAQ. Model or citation failures show the original
policy guidance directly. Expanded retail policies and contacts are fictional.
