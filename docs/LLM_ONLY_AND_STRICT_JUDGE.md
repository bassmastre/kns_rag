# Forced-answer LLM-only baseline and strict LLM judge

This branch keeps the existing token-budget RAG generation pipeline unchanged and
adds two separate paper-facing utilities.

## 1. Forced-answer LLM-only baseline

The baseline uses the same `llm.generator` configuration as Stage 07 but sends
only the QA question. No retrieved context, chunk, evidence ID, reference answer,
or metadata is provided to the generator.

The model is explicitly required to commit to its best answer. Refusals,
`INSUFFICIENT_CONTEXT`, placeholders, and instructions to consult another
document are prohibited. This is the only no-context baseline used in the paper.

```bash
python scripts/07_generate_llm_only.py
```

Default output:

```text
outputs/generation/answers_llm_only_forced.jsonl
```

The earlier abstention-enabled output, if present at
`outputs/generation/answers_llm_only.jsonl`, is not reused and should not be
reported as the baseline result.

The script checkpoints every 10 records and resumes completed
`(experiment_id, generator_model)` pairs by default.

## 2. Strict API LLM judge

The strict evaluator uses the binary O/X prompt supplied for the experiment. It
passes only the question, reference answer, `evidence_keywords`,
`source_section`, and generated answer. Retrieval context is intentionally not
shown to the judge.

RAG answers:

```bash
python scripts/08_eval_llm_judge.py --answers-kind rag
```

Forced-answer LLM-only answers:

```bash
python scripts/08_eval_llm_judge.py --answers-kind llm_only
```

The API/model is read from `llm.judge` in `config.yaml`. The existing
`openai_compatible` backend supports Ollama, vLLM, LM Studio, and hosted
OpenAI-compatible chat-completions endpoints.

Outputs include JSONL detail records, detail CSV, summary JSON, and summary CSV.
Summaries report O/X counts and accuracy overall and by strategy, token budget,
QA type, and source section. Malformed or failed judge calls are stored as
`judge_error` and retried on the next resumable run; they are not silently
converted to X.

The earlier `scripts/08_eval_answers.py` 0–2 point diagnostic rubric remains
available, but the binary strict script is the final evaluator for the reported
O/X accuracy.
