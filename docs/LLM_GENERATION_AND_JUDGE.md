# LLM generation and strict LLM-judge evaluation

## Why both baselines are needed

Use the same generator settings for:

1. `rag`: answers from retrieved context.
2. `llm_only`: no-context closed-book baseline.

The strict judge applies the same O/X rubric to both. A later gold-context run can be produced by preparing a RAG input file containing only the QA's gold evidence.

## Configuration

`config.yaml` contains two roles under `llm`:

- `generator`: normally a local Ollama model through its OpenAI-compatible endpoint.
- `judge`: an external OpenAI-compatible or Anthropic API.

API keys are read only from environment variables. Do not write keys into `config.yaml`.

Example local generator:

```yaml
llm:
  generator:
    provider: openai_compatible
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    name: gemma4:12b
```

For local Ollama, the client uses the ignored placeholder key `ollama` when the configured environment variable is absent.

Example OpenAI-compatible judge:

```yaml
  judge:
    provider: openai_compatible
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    name: <judge-model>
```

Example Anthropic judge:

```yaml
  judge:
    provider: anthropic
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    name: <judge-model>
```

## Generate answers

```powershell
# Existing RAG inputs
python scripts/07_generate_answers.py --mode rag

# Closed-book baseline: 51 QA records, no retrieved context
python scripts/07_generate_answers.py --mode llm_only
```

Both commands checkpoint after every record and resume completed records. Use `--no-resume` to start over.

## Grade answers

```powershell
# RAG answers
python scripts/08_eval_llm_judge.py `
  --answers outputs/generation/answers_rag.jsonl

# LLM-only baseline
python scripts/08_eval_llm_judge.py `
  --answers outputs/generation/answers_llm_only.jsonl `
  --out outputs/eval/llm_only_judge_results.jsonl
```

The judge writes:

- record-level JSONL with verdict, reason, raw judge output, timing, and usage;
- summary JSON grouped by strategy, QA type, and source section;
- compact CSV for inspection.

Transport errors and malformed judge output are recorded as judge errors, not silently converted into X. Rerunning the command retries those records while preserving completed verdicts.
