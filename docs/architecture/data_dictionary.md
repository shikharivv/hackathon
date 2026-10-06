# Data Dictionary

## 1. Policy Documents Schema

Source files stored in `docs/policies/`.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `filename` | string | Original file name | `leave_policy_2024.md` |
| `file_path` | string | Relative path from project root | `docs/policies/leave_policy_2024.md` |
| `format` | string | File extension (pdf, md, txt) | `md` |
| `title` | string | Extracted document title | `Annual Leave Policy` |
| `category` | string | Policy category | `leave`, `benefits`, `conduct`, `compensation` |
| `effective_date` | string (ISO 8601) | Date the policy became effective | `2024-01-15` |
| `version` | string | Document version identifier | `2.1` |
| `last_modified` | string (ISO 8601) | File last-modified timestamp | `2024-06-01T14:30:00Z` |
| `size_bytes` | integer | File size in bytes | `14520` |

## 2. Vector Store Schema

ChromaDB collection: `hr_policies`
Persistence path: `data/vectorstore/`

### 2.1 Document Record

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string (UUID) | Unique chunk identifier | `a1b2c3d4-e5f6-7890-abcd-ef1234567890` |
| `document` | string | Raw text content of the chunk | `"Employees are entitled to 20 days..."` |
| `embedding` | float[384] | Dense vector from sentence-transformers | `[0.023, -0.118, 0.045, ...]` |
| `metadata` | object | Associated metadata (see below) | `{...}` |

### 2.2 Chunk Metadata

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `source` | string | Source file path | `docs/policies/leave_policy_2024.md` |
| `chunk_index` | integer | Position of chunk within the document (0-based) | `3` |
| `total_chunks` | integer | Total number of chunks from this document | `12` |
| `section` | string | Nearest section heading (if available) | `Sick Leave Entitlement` |
| `category` | string | Policy category | `leave` |
| `effective_date` | string | Policy effective date | `2024-01-15` |
| `char_start` | integer | Character offset of chunk start in original document | `2048` |
| `char_end` | integer | Character offset of chunk end in original document | `2560` |
| `chunk_size` | integer | Number of characters in this chunk | `512` |

## 3. API Request / Response Schemas

### 3.1 Query Request

`POST /query`

```json
{
  "question": "How many vacation days do full-time employees get?",
  "top_k": 5,
  "category_filter": "leave",
  "confidence_threshold": 0.65
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | -- | Natural language question |
| `top_k` | integer | No | 5 | Number of chunks to retrieve |
| `category_filter` | string | No | null | Restrict to a policy category |
| `confidence_threshold` | float | No | 0.65 | Minimum similarity score |

### 3.2 Query Response

```json
{
  "answer": "Full-time employees are entitled to 20 vacation days per calendar year.",
  "confidence": 0.87,
  "sources": [
    {
      "source": "docs/policies/leave_policy_2024.md",
      "section": "Annual Leave Entitlement",
      "chunk_index": 2,
      "score": 0.91
    },
    {
      "source": "docs/policies/leave_policy_2024.md",
      "section": "Eligibility",
      "chunk_index": 0,
      "score": 0.78
    }
  ],
  "query_time_ms": 342,
  "model": "gpt-4o-mini",
  "disclaimer": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer text |
| `confidence` | float | Aggregate confidence score (average of source scores) |
| `sources` | array | List of source references |
| `sources[].source` | string | File path of the source document |
| `sources[].section` | string | Section heading within the source |
| `sources[].chunk_index` | integer | Chunk position in the document |
| `sources[].score` | float | Cosine similarity score |
| `query_time_ms` | integer | Total processing time in milliseconds |
| `model` | string | LLM model used for generation |
| `disclaimer` | string or null | Warning if confidence is low or context is insufficient |

### 3.3 Health Check

`GET /health`

```json
{
  "status": "healthy",
  "vectorstore_ready": true,
  "document_count": 156,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_provider": "openai"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` or `unhealthy` |
| `vectorstore_ready` | boolean | Whether the vector store is loaded and accessible |
| `document_count` | integer | Number of indexed chunks |
| `embedding_model` | string | Active embedding model name |
| `llm_provider` | string | Active LLM provider |

### 3.4 Index Request

`POST /index`

```json
{
  "source_dir": "docs/policies",
  "force_reindex": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source_dir` | string | No | `docs/policies` | Directory containing policy documents |
| `force_reindex` | boolean | No | false | Delete existing index and rebuild |

### 3.5 Index Response

```json
{
  "status": "completed",
  "documents_processed": 12,
  "chunks_created": 156,
  "duration_seconds": 8.4
}
```

## 4. Evaluation Metrics Definitions

### 4.1 Test Case Schema

Stored in `tests/fixtures/eval_dataset.json`.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `question` | string | Test question | `"What is the parental leave duration?"` |
| `ground_truth` | string | Expected correct answer | `"16 weeks for primary caregiver..."` |
| `contexts` | array[string] | Expected relevant context passages | `["Section 4.2: Parental leave..."]` |

### 4.2 Evaluation Output Schema

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `faithfulness` | float | 0.0 - 1.0 | Fraction of answer claims supported by retrieved context |
| `answer_relevance` | float | 0.0 - 1.0 | Semantic similarity between answer and question intent |
| `context_precision` | float | 0.0 - 1.0 | Fraction of retrieved chunks relevant to the question |
| `context_recall` | float | 0.0 - 1.0 | Fraction of ground-truth claims covered by retrieved chunks |
| `overall_score` | float | 0.0 - 1.0 | Weighted average of all metrics |

### 4.3 Evaluation Summary

```json
{
  "timestamp": "2024-06-15T10:30:00Z",
  "test_cases": 50,
  "metrics": {
    "faithfulness": {"mean": 0.92, "std": 0.05, "min": 0.78},
    "answer_relevance": {"mean": 0.88, "std": 0.07, "min": 0.71},
    "context_precision": {"mean": 0.85, "std": 0.08, "min": 0.65},
    "context_recall": {"mean": 0.87, "std": 0.06, "min": 0.72}
  },
  "config": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
    "llm_model": "gpt-4o-mini"
  }
}
```
