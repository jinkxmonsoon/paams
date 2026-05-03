# LLM Backend Smoke Test Setup (`btom_v2.runner_llm_real_smoke`)

This guide explains how to run **small real/local smoke tests** for the btom_v2 LLM adapter backends.

> Scope: smoke tests only (not full experiments).

## Runner

```bash
python -m btom_v2.runner_llm_real_smoke --backend <mock|groq|ollama|transformers_local>
```

Default smoke config in runner:
- Scenario: `C5b_costly_false_belief`
- Policies: `LLMReactiveReal,LLMBToMReal`
- Seeds: `0`
- Total episodes: `2`

---

## 1) Groq backend

### Environment variable

```bash
export GROQ_API_KEY="<your_key_here>"
```

### Optional install
(Only needed if your Python environment is missing TLS/HTTP support beyond stdlib; runner uses stdlib HTTP.)

```bash
python -m pip install --upgrade certifi
```

### Example command

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend groq \
  --model llama-3.1-8b-instant \
  --max-llm-calls 20 \
  --max-steps 30 \
  --max-tokens 128
```

### Recommended smallest/low-cost smoke command

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend groq \
  --model llama-3.1-8b-instant \
  --seeds 0 \
  --max-llm-calls 10 \
  --max-steps 20 \
  --max-tokens 96
```

### Expected skip message
- `SKIP_REAL_LLM_SMOKE_NO_GROQ_API_KEY`

---

## 2) Ollama backend

### Prerequisites
Install Ollama and ensure daemon is running locally.

- Linux/macOS: <https://ollama.com/download>
- Verify daemon endpoint:

```bash
curl http://localhost:11434/api/tags
```

### List installed models

```bash
ollama list
```

### Example command

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend ollama \
  --model llama3.1:8b \
  --max-llm-calls 20 \
  --max-steps 30 \
  --max-tokens 128
```

### Recommended smallest smoke command
(Use a model you already have locally.)

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend ollama \
  --model gemma2:2b \
  --seeds 0 \
  --max-llm-calls 10 \
  --max-steps 20 \
  --max-tokens 96
```

### Expected skip message
- `SKIP_REAL_LLM_SMOKE_OLLAMA_UNAVAILABLE`

---

## 3) `transformers_local` backend

### Install commands

```bash
python -m pip install --upgrade pip
python -m pip install "torch" "transformers" "accelerate"
```

### Local-cache-only behavior
By default, the runner uses local cache only for this backend.
If model files are not cached and `--allow-download` is **not** set, setup will skip.

### Example command (cached model)

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend transformers_local \
  --model google/gemma-2-2b \
  --max-llm-calls 20 \
  --max-steps 30 \
  --max-tokens 128
```

### Allow download explicitly

```bash
python -m btom_v2.runner_llm_real_smoke \
  --backend transformers_local \
  --model google/gemma-2-2b \
  --allow-download \
  --max-llm-calls 20 \
  --max-steps 30 \
  --max-tokens 128
```

### Expected skip messages
- `SKIP_REAL_LLM_SMOKE_TRANSFORMERS_UNAVAILABLE`
- `SKIP_REAL_LLM_SMOKE_TRANSFORMERS_MODEL_LOAD_FAILED`

---

## Suggested low-cost models for smoke tests

- Groq: `llama-3.1-8b-instant`
- Ollama (local): `gemma2:2b` (if installed), `llama3.1:8b` (if already present)
- Transformers local: small cached instruction/chat model (e.g. 2B-ish class) already available in local cache

> Always prefer a model already installed/cached to avoid delays/cost.

---

## Budget warnings (important)

Use tight caps for smoke:
- `--max-llm-calls 10..30`
- `--max-steps 20..40`
- `--max-tokens 64..128`

Why:
- Each agent step can trigger a model call.
- High caps can grow token usage quickly.
- This runner is for connectivity/plumbing validation, not quality benchmarking.

---

## Troubleshooting

### Missing Groq API key
Symptom:
- `SKIP_REAL_LLM_SMOKE_NO_GROQ_API_KEY`

Fix:
```bash
export GROQ_API_KEY="<your_key_here>"
```
Re-run same command.

### Ollama unavailable
Symptoms:
- `SKIP_REAL_LLM_SMOKE_OLLAMA_UNAVAILABLE`
- `curl .../api/tags` fails
- `ollama list` not found

Fix:
1. Install Ollama.
2. Start Ollama service/daemon.
3. Confirm `http://localhost:11434/api/tags` works.
4. Ensure chosen model exists in `ollama list`.

### Missing `transformers`/`torch`
Symptom:
- `SKIP_REAL_LLM_SMOKE_TRANSFORMERS_UNAVAILABLE`

Fix:
```bash
python -m pip install torch transformers accelerate
```

### Model not in local cache (`transformers_local`)
Symptom:
- `SKIP_REAL_LLM_SMOKE_TRANSFORMERS_MODEL_LOAD_FAILED`

Fix options:
1. Use a model already in cache.
2. Re-run with `--allow-download` if you intentionally want download.

---

## Minimal end-to-end checklist

1. Mock sanity:
```bash
python -m btom_v2.runner_llm_real_smoke --backend mock
```
2. Groq check:
```bash
python -m btom_v2.runner_llm_real_smoke --backend groq --model llama-3.1-8b-instant
```
3. Ollama availability:
```bash
curl http://localhost:11434/api/tags
ollama list
```
4. Ollama smoke if available:
```bash
python -m btom_v2.runner_llm_real_smoke --backend ollama --model <installed_model> --max-llm-calls 20 --max-steps 30
```

