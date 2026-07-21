# Downstream QA evaluation

This pipeline compares chunking strategies at equal retrieved-context token budgets.

## Stages

```text
04_retrieve.py          dense ranking + cumulative token counts
05_build_rag_inputs.py  one input per QA × strategy × token budget
07_generate_answers.py  context-grounded answer generation
08_eval_answers.py      reference-based LLM judge + aggregate metrics
```

`06_eval_retrieval.py` remains the retrieval-only evaluation stage.

## Configure the models

The generator and judge are configured independently in `config.yaml`.
Using different models is preferable. If the same model is used for both roles,
report that limitation in the paper.

### Local Hugging Face model

```yaml
llm:
  generator:
    mode: transformers
    name: /path/to/generator-model
    max_new_tokens: 256
    temperature: 0.0
    device_map: auto
    torch_dtype: auto
  judge:
    mode: transformers
    name: /path/to/judge-model
    max_new_tokens: 384
    temperature: 0.0
    device_map: auto
    torch_dtype: auto
```

Install local-generation dependencies:

```bash
pip install -e ".[all]"
```

### OpenAI-compatible local or hosted server

```yaml
llm:
  generator:
    mode: openai_compatible
    name: generator-model-name
    base_url: http://localhost:8000/v1
    api_key_env: null
  judge:
    mode: openai_compatible
    name: judge-model-name
    base_url: http://localhost:8001/v1
    api_key_env: null
```

For an authenticated endpoint, set `api_key_env` to the name of an environment
variable and set the secret only in the process environment. Do not store the key
in `config.yaml`.

## Recommended pilot

Run one budget and a small subset before the full experiment:

```bash
python scripts/05_build_rag_inputs.py --token-budget 512
python scripts/07_generate_answers.py --token-budget 512 --limit 20
python scripts/08_eval_answers.py --token-budget 512 --limit 20
```

Inspect:

```text
outputs/generation/answers.jsonl
outputs/eval/answer_judgements.jsonl
outputs/eval/downstream_metrics.csv
```

Then run all 75 questions and five strategies at 512 tokens:

```bash
python scripts/07_generate_answers.py --token-budget 512
python scripts/08_eval_answers.py --token-budget 512
```

To compare all configured budgets, rebuild Stage 05 without an override and run
Stages 07 and 08 without budget filters:

```bash
python scripts/05_build_rag_inputs.py
python scripts/07_generate_answers.py
python scripts/08_eval_answers.py
```

With 75 questions, five strategies, and three budgets, this creates 1,125 answers
and 1,125 judge evaluations. Checkpointing and resume are enabled by default.
Use `--no-resume` only when intentionally replacing prior results.

## Main downstream metrics

- `pass_rate`: strict pass; all three rubric dimensions must receive 2/2 and no unsupported claim may be present.
- `correctness`: normalized factual-correctness score.
- `completeness`: normalized coverage of all answer elements required by the question.
- `relation_accuracy`: normalized accuracy of Condition–Action, Action–Completion-Time, AND/OR, notes, and alternatives.
- `unsupported_claim_rate`: fraction of answers containing unsupported content.
- `mean_lexical_f1`: secondary lexical-overlap diagnostic only.

The judge also records these error labels:

```text
wrong_condition
wrong_action
wrong_completion_time
connector_error
missing_required_action
unsupported_claim
insufficient_context
other
```

The paper should report `pass_rate`, `relation_accuracy`, and error-type counts by
strategy, token budget, and QA type. Retrieval metrics and downstream metrics must
be presented separately.
