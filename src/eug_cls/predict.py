"""Generate classifier predictions (greedy) and parse the JSON label.

Input: data/classifier_v1/eval_{val,test}.jsonl (prompt = [system, user] messages, gold JSON + metadata).
Output: one row per example with the raw output, parsed prediction and the gold metadata, plus a
<out>.meta.json sidecar recording the setup. Score with `python -m eug_cls.score`.

Few-shot (the fair baseline for untrained models, CLAUDE.md point 4): --few-shot K inserts K fixed
train examples as user/assistant turns between the system prompt and the test prompt. Exemplars are
label-balanced, cover different categories and languages, are short, come from the TRAIN split only
(no meta_id overlap with val/test) and are identical for every test prompt.

Usage:
    # trained model (merged weights) on the test split, vLLM on GPU
    python -m eug_cls.predict --model runs/aya8b_dpo_hardneg/merged --out runs/eval/aya8b_dpo_hardneg/test.jsonl
    # few-shot baseline
    python -m eug_cls.predict --model CohereLabs/aya-expanse-8b --few-shot 6 --out runs/eval/aya8b_fewshot6/test.jsonl
    # CPU check with transformers
    python -m eug_cls.predict --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --limit 20 --out /tmp/p.jsonl

Parsing (`parse` field): strict = the whole output is one valid JSON object with valid fields
(and category_id null when label is SAFE, as the system prompt requires);
lenient = a valid label/category could be extracted (JSON inside fences/extra text, or key-value
regex); invalid = nothing usable. Scoring uses strict+lenient predictions; the strict rate is reported
separately, so format errors stay visible.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

LABELS = ("SAFE", "EDGE_CASE", "VIOLATION")
CATEGORIES = tuple(f"G{i}" for i in range(1, 8))
KEEP_META = ("example_id", "concept_id", "meta_id", "split", "language", "label", "category_id",
             "source_group", "scenario_type", "augmentation", "source_run", "generator")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- parsing
def _valid(label, cat) -> bool:
    return label in LABELS and (cat is None or cat in CATEGORIES)


def _norm_cat(c):
    if c is None:
        return None
    c = str(c).strip().upper()
    return None if c in ("", "NULL", "NONE") else c


def parse_output(text: str) -> tuple[str | None, str | None, str]:
    """(label, category_id, parse) with parse in strict | lenient | invalid."""
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and set(obj) == {"label", "category_id"}:
            lab, cat = obj["label"], obj["category_id"]
            if _valid(lab, cat):
                # the system prompt requires category_id null for SAFE; anything else is not a clean answer
                return lab, cat, "strict" if (lab != "SAFE" or cat is None) else "lenient"
    except (json.JSONDecodeError, TypeError):
        pass
    # any JSON object inside the text (code fences, prose around it)
    for m in re.finditer(r"\{[^{}]*\}", t, flags=re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "label" in obj:
            lab = str(obj["label"]).strip().upper()
            cat = _norm_cat(obj.get("category_id"))
            if _valid(lab, cat):
                return lab, cat, "lenient"
    # key-value regex fallback (e.g. single quotes, missing braces)
    ml = re.search(r"label\W{0,4}(SAFE|EDGE_CASE|VIOLATION)", t, flags=re.I)
    if ml:
        mc = re.search(r"category(?:_id)?\W{0,4}(G[1-7]|null|none)", t, flags=re.I)
        cat = _norm_cat(mc.group(1)) if mc else None
        return ml.group(1).upper(), cat, "lenient"
    return None, None, "invalid"


# ---------------------------------------------------------------- few-shot
def pick_exemplars(train_file: str, k: int, seed: int) -> list[dict]:
    """k short train examples, cycling SAFE / EDGE_CASE / VIOLATION, distinct categories and languages."""
    rows = read_jsonl(train_file)
    rows = [r for r in rows if r["split"] == "train"]
    lens = sorted(len(r["messages"][1]["content"]) for r in rows)
    short = [r for r in rows if len(r["messages"][1]["content"]) <= lens[len(lens) // 4]]  # shortest quartile
    rng = random.Random(seed)
    rng.shuffle(short)
    picked, langs, cats = [], set(), set()
    order = [LABELS[i % 3] for i in range(k)]
    for lab in order:
        for r in short:
            if r["label"] != lab or r["language"] in langs or r in picked:
                continue
            if lab != "SAFE" and r["category_id"] in cats and len(cats) < 7:
                continue
            picked.append(r)
            langs.add(r["language"])
            cats.add(r["category_id"] or r["source_group"])
            break
    rng.shuffle(picked)
    return picked


def with_few_shot(prompt: list[dict], shots: list[dict]) -> list[dict]:
    if not shots:
        return prompt
    turns = []
    for s in shots:
        turns += [s["messages"][1], s["messages"][2]]
    assert prompt[0]["role"] == "system"
    return [prompt[0], *turns, *prompt[1:]]


# ---------------------------------------------------------------- generation
def gen_hf(model_path, convs, max_new_tokens, batch_size, dtype, adapter=None, load_in_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": getattr(torch, dtype)}
    if load_in_4bit:   # same NF4 setup as training with model.load_in_4bit
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype))
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if not load_in_4bit:
        model = model.to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)   # top-level ("default") adapter = the trained one
    model.eval()
    for i in range(0, len(convs), batch_size):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in convs[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            yield tok.decode(seq, skip_special_tokens=True).strip()
        print(f"  generated {min(i + batch_size, len(convs))}/{len(convs)}", flush=True)


def gen_vllm(model_path, convs, max_new_tokens, tp, max_model_len):
    from vllm import LLM, SamplingParams  # GPU box only

    llm = LLM(model=model_path, tensor_parallel_size=tp, max_model_len=max_model_len, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    for o in llm.chat(convs, params):
        yield o.outputs[0].text.strip()


def lib_versions(names=("torch", "transformers", "vllm")) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local dir (use <run>/merged for trained runs)")
    ap.add_argument("--data", default="data/classifier_v1/eval_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--few-shot-pool", default="data/classifier_v1/sft_train.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=48, help="gold answers are ~15 tokens")
    ap.add_argument("--batch-size", type=int, default=8, help="hf backend only")
    ap.add_argument("--dtype", default="bfloat16", help="hf backend only (float32 on CPU, float16 on T4)")
    ap.add_argument("--adapter", help="hf backend: LoRA adapter dir or hub id applied on top of --model")
    ap.add_argument("--load-in-4bit", action="store_true", help="hf backend: NF4 base model (QLoRA runs)")
    ap.add_argument("--tp", type=int, default=1, help="vllm tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, help="first N examples (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = read_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    shots = pick_exemplars(a.few_shot_pool, a.few_shot, a.seed) if a.few_shot else []
    shot_metas = {s["meta_id"] for s in shots}
    if shot_metas & {r["meta_id"] for r in rows}:
        raise SystemExit("few-shot exemplar shares a meta_id with the evaluated data")
    convs = [with_few_shot(r["prompt"], shots) for r in rows]

    t0 = time.time()
    if a.backend == "vllm":
        if a.adapter or a.load_in_4bit:
            raise SystemExit("--adapter / --load-in-4bit are hf-backend options (vLLM: pass merged weights)")
        outs = list(gen_vllm(a.model, convs, a.max_new_tokens, a.tp, a.max_model_len))
    else:
        outs = list(gen_hf(a.model, convs, a.max_new_tokens, a.batch_size, a.dtype, a.adapter, a.load_in_4bit))
    secs = time.time() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"strict": 0, "lenient": 0, "invalid": 0}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r, text in zip(rows, outs):
            lab, cat, how = parse_output(text)
            counts[how] += 1
            f.write(json.dumps({**{k: r.get(k) for k in KEEP_META}, "gold": r["gold"], "output": text,
                                "pred_label": lab, "pred_category": cat, "parse": how},
                               ensure_ascii=False) + "\n")
    meta = {"model": a.model, "adapter": a.adapter, "load_in_4bit": a.load_in_4bit, "data": a.data,
            "n": len(rows), "backend": a.backend, "few_shot": a.few_shot,
            "few_shot_example_ids": [s["example_id"] for s in shots],
            "few_shot_languages": [s["language"] for s in shots],
            "max_new_tokens": a.max_new_tokens, "decoding": "greedy", "seconds": round(secs, 1),
            "parse_counts": counts, "versions": lib_versions()}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), **{k: meta[k] for k in ("n", "seconds", "parse_counts")}}))


if __name__ == "__main__":
    main()
