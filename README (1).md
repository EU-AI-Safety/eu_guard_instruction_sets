# EU-Guard

EU-Guard is a research project for generating and curating datasets to train and evaluate an AI safety classifier against the project's seven-group EU AI Act prohibited-practice taxonomy (`G1`–`G7`). The repository includes meta-prompt generation, synthetic example generation, optional external-data curation, SFT/DPO formatting, splitting, and training utilities.

> **Legal note:** The scripts encode a project-specific plain-language taxonomy. Before publication or compliance use, map every group, exception, and edge-case rule to the authoritative text of Regulation (EU) 2024/1689 and obtain subject-matter review. Historical script versions may use inconsistent group descriptions; maintain one canonical taxonomy across generation, validation, and training.

## 1. End-to-end workflow

```text
Project EU AI Act taxonomy and rule descriptions
                    |
                    v
       Meta-prompt generation script
                    |
                    v
        eu_ai_act_dataset_v3.jsonl
                    |
                    v
       Generation LLM through API
                    |
                    v
       Structured synthetic examples
                    |
          parsing / validation
             / deduplication
                    |
             +------+------+
             |             |
             v             v
           SFT           DPO
             |             |
             +------+------+
                    |
              train / val / test
```

There are two related but distinct data paths:
1. A synthetic path based on the project's own EU AI Act meta-prompts.
2. An optional mixed-data curation path that also uses external safety datasets (Section 8).

## 2. Meta-prompt / instruction-set generation

The meta-prompt generator (project files include `eu_ai_act_metaprompts.py` and later variants) creates **instructions for a downstream generation model**. A meta-prompt is a generation recipe, not the final user request used to train the classifier.

The inspected generator uses:
- Seven project prohibition groups (`G1`–`G7`).
- Three graduated levels: `VIOLATION`, `EDGE_CASE`, and `COMPLIANT`.
- Four persona pools: vendor, researcher, government, and legal.
- Randomized wording, output types, and scenario framing to create lexical variety while retaining the underlying rule.
- Red-team framing, euphemisms, and purported safeguards (including safeguards that do not necessarily remove the underlying concern).
- Self-contained plain-language rule blocks embedded in each prompt, rather than relying only on article numbers.
- Structured output instructions so generated examples can be parsed and audited.

The `eu_ai_act_dataset_v3.jsonl` run discussed in this project contains **1,680 meta-prompt records**. Records include fields such as `id`, `group`, `group_label`, `level`, language fields, `persona_tone`, `augmentation`, `variant_index`, `meta_prompt`, `expected_behavior`, and `tags`. The project run describes coverage across 24 language codes, four personas, and eight augmentation settings. Verify exact counts against the particular JSONL file and its distribution report for each release.

### Legal/taxonomy source

The meta-prompt code contains the plain-language rule blocks used as the operational taxonomy. The inspected code does not provide a complete article-by-article legal citation map for every rule and exception. Therefore, this README does not claim that every prompt has been legally validated against a specific article. For a release, document the exact EU AI Act provisions used for each group, the interpretation of exceptions, and the reviewer/sign-off.

## 3. Synthetic user-request generation

The Qwen SFT/DPO generator reads the `meta_prompt` field from each record in `eu_ai_act_dataset_v3.jsonl` and sends it to a generation model through an OpenAI-compatible Chat Completions API (for example, OpenRouter).

The generation model is instructed to **generate realistic user requests, not to act as the final classifier**. Its structured response contains:
- `prompt`: the actual user request used as classifier input.
- `label`: `VIOLATION`, `EDGE_CASE`, or `SAFE`.
- `category_id`: `G1`–`G7`, or `null` for `SAFE`.
- `scenario_type`, `context`, `target_behavior`, and `rationale`: supporting audit metadata, not part of the final user message.

Generation instructions require category fidelity, realistic and independently understandable requests, variation in industries, motivations and phrasing, and avoidance of repeated template-like prompts. They prohibit copying the meta-prompt into the generated user request or placing the classifier answer inside that request.

The model is selected through a command-line argument. Models used in the project runs discussed include:
- `qwen/qwen3-32b`
- `google/gemini-3.1-flash-lite`

The scripts use the OpenAI Python client against an OpenAI-compatible API endpoint. Record the exact model/provider and run parameters for every generated dataset.

## 4. Validation and deduplication

The inspected generator performs programmatic checks including:
- non-empty prompt;
- allowed label and category values;
- `SAFE` examples must have a null category;
- non-safe examples must have a valid category;
- configured prompt-length limits;
- rejection of meta-prompt leakage;
- exact-prompt deduplication using a content hash.

An optional independent model validator can assess whether the assigned label/category is supported and whether an example is realistic and specific. This option is not necessarily enabled in every run; record whether it was enabled and which validator model was used.

Audit fields are retained separately in `metadata.jsonl`. The classifier SFT conversation contains only the classifier system instruction, generated user request, and target assistant output.

## 5. SFT format

The Qwen SFT output is chat-style JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "EU-Guard classifier instructions and taxonomy..."},
    {"role": "user", "content": "Generated user request..."},
    {"role": "assistant", "content": "{\"label\":\"VIOLATION\",\"category_id\":\"G4\"}"}
  ]
}
```

The assistant target is a compact JSON string with `label` and `category_id`. The system prompt asks the classifier to return only valid JSON, using `SAFE`, `EDGE_CASE`, or `VIOLATION`. For `SAFE`, the category is `null`.

Typical outputs:
- `qwen_all.jsonl`
- `qwen_train.jsonl`
- `qwen_val.jsonl`
- `qwen_test.jsonl`
- `metadata.jsonl`
- `stats.json`

## 6. DPO construction

The DPO files in the inspected Qwen generator are **derived programmatically from the same synthetic examples and their assigned labels**. They are not taken from a separate human preference dataset, and chosen/rejected responses are not human preference annotations.

For each example:
- `prompt` is the generated user request.
- `chosen` is the classifier JSON corresponding to the assigned label/category.
- `rejected` is a deliberately incorrect alternative created by a rule in the script.

Current rejected-answer policy:

| Assigned label | Chosen | Rejected |
|---|---|---|
| `VIOLATION` | `VIOLATION` + its category | `SAFE` + null category |
| `SAFE` | `SAFE` + null category | `VIOLATION` + a category (the code uses `G1` if no category exists) |
| `EDGE_CASE` | `EDGE_CASE` + relevant category | `SAFE` + null category |

Example DPO record:

```json
{
  "prompt": "Generated user request...",
  "chosen": "{\"label\":\"VIOLATION\",\"category_id\":\"G4\"}",
  "rejected": "{\"label\":\"SAFE\",\"category_id\":null}"
}
```

Outputs:
- `dpo_train.jsonl`
- `dpo_val.jsonl`
- `dpo_test.jsonl`

**DPO limitation:** These are synthetic hard-negative pairs derived from labels, not independently collected human rankings. In particular, the current `EDGE_CASE` rule rejects `SAFE`; this is a scripted contrast, not a claim that every borderline case should be treated as unsafe. Review this policy before training. A stronger preference dataset would use human-reviewed or independently adjudicated alternatives and more than one carefully designed negative where appropriate.

## 7. Splits and leakage considerations

The generator defaults to:
- **90% train** — parameter learning.
- **5% validation** — training-time evaluation/model selection.
- **5% test** — held-out final evaluation.

The script creates the SFT split and then builds DPO files from the corresponding train, validation, and test lists, keeping SFT/DPO membership aligned for that run.

The split function groups rows by `prompt_id` before assigning groups to splits. However, the inspected `assign_prompt_ids()` creates the ID using `meta_id`, augmentation, variant index **and `example_index`**. Because the example index is included, IDs are specific to individual generated examples. Thus, the current code does **not guarantee** that all examples from one source meta-prompt or scenario family stay in the same split. Exact deduplication also does not catch every semantic or translated near-duplicate.

For a strict leakage-resistant benchmark, assign a stable `scenario_family_id` shared by related examples, paraphrases, augmentations, and language variants; split by that family ID; then run exact and semantic near-duplicate checks across splits. Keep the test set untouched during model selection.

## 8. Optional mixed-dataset curation using external sources

The repository also contains a separate mixed-data builder (for example, `build_eu_guard_mixed_v2.py` and later three-way/quota-fill versions). This path is distinct from the synthetic Qwen SFT/DPO generator.

The inspected external-candidate extraction script uses:
1. **Anthropic HH-RLHF** (`Anthropic/hh-rlhf`), attempting the `red-team-attempts` configuration and falling back to the default dataset when needed. It extracts the human/user portion of conversation text as candidate prompts.
2. **PKU-Alignment BeaverTails** (`PKU-Alignment/BeaverTails`), extracting prompt text and retaining available response, safety target, category, and source metadata.

These external prompts are initially marked as needing EU AI Act annotation. Their original labels/categories are not automatically treated as EU AI Act labels.

The mixed builder uses the project's EU synthetic prompts and plain-language rule descriptions as category anchors. The inspected implementation uses Sentence Transformers (default `sentence-transformers/all-MiniLM-L6-v2`) to embed anchors and candidate prompts, then applies semantic similarity, top-k category matching, score thresholds/margins, per-category quotas, deduplication, and source-provenance tracking. Three-way variants distinguish high-confidence violation matches, medium-confidence candidates/edge cases, and safe-region candidates.

This is a **semantic candidate-mapping/curation pipeline, not a legal annotation oracle**. The builder itself notes that mappings should be audited. Review mapped examples against the legal taxonomy before training.

The external HH-RLHF and BeaverTails sources belong to this optional mixed-data branch. They are **not the source of chosen/rejected labels in the synthetic DPO construction in Section 6**.

## 9. Checkpoints and resumability

The parallel generator supports concurrent API workers and checkpoint-based resumption. It writes generated examples to `checkpoint_examples.jsonl` and completed meta-prompt state to `checkpoint_state.json` in the output directory. Resume with `--resume`, using the same output directory and compatible run settings. Keep checkpoints until final outputs and counts have been verified.

## 10. Common artifacts

| File | Purpose |
|---|---|
| `eu_ai_act_dataset_v3.jsonl` | Structured meta-prompt instruction set |
| `qwen_all.jsonl` | All accepted SFT examples |
| `qwen_train.jsonl` | SFT training split |
| `qwen_val.jsonl` | SFT validation split |
| `qwen_test.jsonl` | SFT test split |
| `dpo_train.jsonl` | DPO training preference pairs |
| `dpo_val.jsonl` | DPO validation preference pairs |
| `dpo_test.jsonl` | DPO test preference pairs |
| `metadata.jsonl` | Generation and audit metadata |
| `stats.json` | Dataset counts and run information |
| `checkpoint_examples.jsonl` | Generated-example checkpoint |
| `checkpoint_state.json` | Completed meta-prompt state |

File names can differ across script versions and output folders.

## 11. Reproducibility checklist

For each dataset release, record:
- meta-prompt JSONL version/hash;
- generator and curation script versions or Git commit;
- generation model/provider, date, temperature, token limit, and worker count;
- whether independent model validation was enabled and which model was used;
- counts before/after validation and deduplication;
- label, category, language, and source distributions;
- split seed, ratios, and grouping key;
- exact DPO negative-construction policy;
- manual review, corrections, and exclusions.

EU-Guard is a research/model-development dataset, not legal advice or a definitive regulatory interpretation. Review taxonomy definitions, exceptions, translations, and annotations with qualified subject-matter experts before regulatory or deployment use.
