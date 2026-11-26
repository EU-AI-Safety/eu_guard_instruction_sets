"""Build leak-free SFT / DPO files for the EU-Guard classifier (Path A, see CLAUDE.md).

Steps
1. Read the merged pairs (merged_datasets/dpo_{train,val,test}_combined.jsonl) and join each prompt
   (whitespace/case-normalised) to its row in {qwen_dataset_10k,gemini_dataset_10k,gemini_dataset_20k}/metadata.jsonl.
   Every prompt must join and the merged `chosen` must equal the metadata gold label.
2. Re-split GROUPED BY meta_id (the old random split leaked: all test prompts shared a meta_id with train).
   Each meta_id encodes group-level-language, so the split is stratified by language (exact per-language
   counts) and greedily balanced over the 21 group x level cells.
3. Write, per split, in TRL conversational format with the classifier system prompt in every prompt:
     sft_{split}.jsonl           messages = [system, user, assistant]
     dpo_fixed_{split}.jsonl     original rule-fixed negatives (SAFE -> VIOLATION/G1, else -> SAFE)
     dpo_hardneg_{split}.jsonl   hard negatives (level swap, wrong category, varied SAFE negatives)
     eval_{split}.jsonl          prompt messages + gold, for generation-based evaluation
   Pair files carry the fields required by eu-guard-dpo-main/src/eug_dpo/io.py::check_pair
   (example_id, concept_id=meta_id, split, language, target, pair_type) so they load unchanged there.
4. Write stats.json + README.md (counts, leakage checks, negative-type mix).

Usage: python scripts/build_classifier_data.py [--out data/classifier_v1] [--seed 13]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("qwen_dataset_10k", "gemini_dataset_10k", "gemini_dataset_20k")
SPLITS = ("train", "val", "test")
LABELS = ("SAFE", "EDGE_CASE", "VIOLATION")
GROUPS = tuple(f"G{i}" for i in range(1, 8))
LEVEL_OF = {"COM": "SAFE", "EDG": "EDGE_CASE", "VIO": "VIOLATION"}

# Verbatim system prompt of the existing SFT files (qwen_*.jsonl); checked against them at runtime.
SYSTEM_PROMPT_SOURCE = ROOT / "merged_datasets" / "qwen_train_combined.jsonl"

# hard-negative mix per gold label: (neg_type, probability)
HARD_MIX = {
    "VIOLATION": (("level_swap", 0.4), ("wrong_category", 0.4), ("predict_safe", 0.2)),
    "EDGE_CASE": (("level_swap", 0.4), ("wrong_category", 0.4), ("predict_safe", 0.2)),
    "SAFE": (("flag_source_group", 0.5), ("flag_random_group", 0.5)),
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def label_json(label: str, cat: str | None) -> str:
    # same compact serialisation as the original data
    return json.dumps({"label": label, "category_id": cat}, separators=(",", ":"))


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_system_prompt() -> str:
    prompts = set()
    for i, r in enumerate(read_jsonl(SYSTEM_PROMPT_SOURCE)):
        prompts.add(r["messages"][0]["content"])
        if i > 2000:
            break
    if len(prompts) != 1:
        raise SystemExit(f"expected one system prompt in {SYSTEM_PROMPT_SOURCE}, found {len(prompts)}")
    return prompts.pop()


def load_examples() -> list[dict]:
    meta: dict[str, dict] = {}
    for run in RUNS:
        for r in read_jsonl(ROOT / run / "metadata.jsonl"):
            k = norm(r["prompt"])
            if k in meta:
                raise SystemExit(f"duplicate prompt across metadata files: {r['prompt'][:80]!r}")
            meta[k] = {**r, "source_run": run}
    out, seen = [], set()
    for old_split in ("train", "val", "test"):
        for r in read_jsonl(ROOT / "merged_datasets" / f"dpo_{old_split}_combined.jsonl"):
            k = norm(r["prompt"])
            m = meta.get(k)
            if m is None:
                raise SystemExit(f"merged prompt has no metadata: {r['prompt'][:80]!r}")
            if k in seen:
                raise SystemExit(f"duplicate merged prompt: {r['prompt'][:80]!r}")
            seen.add(k)
            gold = json.loads(r["chosen"])
            if (gold["label"], gold["category_id"]) != (m["label"], m["category_id"]):
                raise SystemExit(f"chosen != metadata gold for meta {m['meta_id']}")
            g, lvl, lang = m["meta_id"].split("-")[:3]
            if (g, LEVEL_OF[lvl], lang) != (m["source_group"], m["label"], m["language_code"]):
                raise SystemExit(f"meta_id does not match metadata fields: {m['meta_id']}")
            out.append({**m, "prompt": r["prompt"], "old_split": old_split,
                        "fixed_rejected": r["rejected"]})
    return out


def label_quotas(n_by_lang_label: dict, frac: float, rng: random.Random) -> dict[tuple[str, str], int]:
    """(language, label) -> number of metas for one held-out split.

    Every language gets floor(n * frac) metas of each label (so every label is present in every
    language), plus extra metas of distinct labels until the language total is round(n_lang * frac) and
    the global label totals are round(n_label * frac)."""
    langs = sorted({l for l, _ in n_by_lang_label})
    q = {k: math.floor(n * frac) for k, n in n_by_lang_label.items()}
    remaining = {lab: round(sum(n for (l, b), n in n_by_lang_label.items() if b == lab) * frac)
                 - sum(v for (l, b), v in q.items() if b == lab) for lab in LABELS}
    order = langs[:]
    rng.shuffle(order)
    for lang in order:
        n_lang = sum(n for (l, _), n in n_by_lang_label.items() if l == lang)
        extra = round(n_lang * frac) - sum(v for (l, _), v in q.items() if l == lang)
        labs = sorted(LABELS, key=lambda b: (-remaining[b], rng.random()))[:extra]
        for b in labs:
            if remaining[b] <= 0 or q[(lang, b)] >= n_by_lang_label[(lang, b)]:
                raise SystemExit(f"cannot meet label quotas for {lang}")
            q[(lang, b)] += 1
            remaining[b] -= 1
    if any(remaining.values()):
        raise SystemExit(f"label quotas not met: {remaining}")
    return q


def split_metas(examples: list[dict], seed: int, frac=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """meta_id -> split. Per (language, label) quotas (label_quotas); within those, metas come from the
    groups currently least represented in that split for that label (random tie-break)."""
    rng = random.Random(seed)
    info = {}
    for e in examples:
        info[e["meta_id"]] = (e["language_code"], e["source_group"], e["label"])
    pools = collections.defaultdict(list)
    for mid, (lang, g, lab) in sorted(info.items()):
        pools[(lang, lab)].append(mid)
    for p in pools.values():
        rng.shuffle(p)
    n_by = {k: len(v) for k, v in pools.items()}
    cell_total = collections.Counter((g, lab) for _, g, lab in info.values())
    assign = {}
    for s, fr in (("test", frac[2]), ("val", frac[1])):
        quotas = label_quotas(n_by, fr, rng)
        cell_count = collections.Counter()
        keys = sorted(quotas)
        rng.shuffle(keys)
        for key in keys:
            pool = pools[key]
            for _ in range(quotas[key]):
                def deficit(mid):
                    c = info[mid][1:]
                    return cell_count[c] / cell_total[c]
                best = min(deficit(m) for m in pool)
                mid = rng.choice([m for m in pool if deficit(m) == best])
                pool.remove(mid)
                assign[mid] = s
                cell_count[info[mid][1:]] += 1
    for pool in pools.values():
        for mid in pool:
            assign[mid] = "train"
    return assign


def hard_negative(e: dict) -> tuple[str, str]:
    """(rejected_json, neg_type); deterministic per prompt (seeded by prompt_hash)."""
    rng = random.Random(int(hashlib.sha256(e["prompt_hash"].encode()).hexdigest()[:16], 16))
    lab, cat = e["label"], e["category_id"]
    types, probs = zip(*HARD_MIX[lab])
    t = rng.choices(types, probs)[0]
    if t == "level_swap":
        return label_json("EDGE_CASE" if lab == "VIOLATION" else "VIOLATION", cat), t
    if t == "wrong_category":
        return label_json(lab, rng.choice([g for g in GROUPS if g != cat])), t
    if t == "predict_safe":
        return label_json("SAFE", None), t
    flag = rng.choice(("EDGE_CASE", "VIOLATION"))
    if t == "flag_source_group":
        return label_json(flag, e["source_group"]), t
    return label_json(flag, rng.choice(GROUPS)), t  # flag_random_group


def build_rows(e: dict, split: str, system: str) -> dict:
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": e["prompt"]}]
    gold = label_json(e["label"], e["category_id"])
    meta = {
        "example_id": e["prompt_hash"][:16],
        "concept_id": e["meta_id"],  # grouping unit for paired bootstrap (reference: concept-level)
        "meta_id": e["meta_id"],
        "split": split,
        "language": e["language_code"],
        "language_name": e["language_name"],
        "label": e["label"],
        "category_id": e["category_id"],
        "source_group": e["source_group"],
        "scenario_type": e.get("scenario_type"),
        "augmentation": e.get("augmentation"),
        "source_run": e["source_run"],
        "generator": "qwen" if e["source_run"].startswith("qwen") else "gemini",
        # reference-schema compatibility: SAFE ~ should_answer, EDGE_CASE/VIOLATION ~ should_refuse
        "target": "should_answer" if e["label"] == "SAFE" else "should_refuse",
        "pair_type": e["label"],
    }
    hard_rej, neg_type = hard_negative(e)
    return {
        "sft": {"messages": prompt + [{"role": "assistant", "content": gold}], **meta},
        "fixed": {"prompt": prompt, "chosen": [{"role": "assistant", "content": gold}],
                  "rejected": [{"role": "assistant", "content": e["fixed_rejected"]}],
                  **meta, "neg_type": "fixed_rule"},
        "hard": {"prompt": prompt, "chosen": [{"role": "assistant", "content": gold}],
                 "rejected": [{"role": "assistant", "content": hard_rej}], **meta, "neg_type": neg_type},
        "eval": {"prompt": prompt, "gold": gold, **meta},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/classifier_v1")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out = ROOT / args.out

    system = load_system_prompt()
    ex = load_examples()
    assign = split_metas(ex, args.seed)

    files = {k: {s: [] for s in SPLITS} for k in ("sft", "fixed", "hard", "eval")}
    for e in sorted(ex, key=lambda e: (e["meta_id"], e["source_run"], e["example_index"], e["prompt_hash"])):
        s = assign[e["meta_id"]]
        for k, row in build_rows(e, s, system).items():
            files[k][s].append(row)
    for k in files:
        for s in SPLITS:
            random.Random(f"{args.seed}-{k}-{s}").shuffle(files[k][s])

    names = {"sft": "sft_{}.jsonl", "fixed": "dpo_fixed_{}.jsonl", "hard": "dpo_hardneg_{}.jsonl",
             "eval": "eval_{}.jsonl"}
    written = {}
    for k, pat in names.items():
        for s in SPLITS:
            if k == "eval" and s == "train":
                continue
            written[pat.format(s)] = write_jsonl(out / pat.format(s), files[k][s])

    # ---- checks + stats ----
    metas = {s: {r["meta_id"] for r in files["sft"][s]} for s in SPLITS}
    texts = {s: {norm(r["messages"][1]["content"]) for r in files["sft"][s]} for s in SPLITS}
    leak = {f"{a}&{b}": {"meta_ids": len(metas[a] & metas[b]), "prompts": len(texts[a] & texts[b])}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(v for d in leak.values() for v in d.values()):
        raise SystemExit(f"leakage after re-split: {leak}")
    for r in files["hard"]["train"]:
        if r["chosen"][0]["content"] == r["rejected"][0]["content"]:
            raise SystemExit("hard negative equals chosen")

    def counter(rows, key):
        return dict(sorted(collections.Counter(key(r) for r in rows).items()))

    stats = {"seed": args.seed, "n_total": len(ex), "files": written, "leakage_check": leak,
             "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest()[:16],
             "hard_negative_mix": {k: dict(v) for k, v in HARD_MIX.items()}, "splits": {}}
    for s in SPLITS:
        rows = files["sft"][s]
        stats["splits"][s] = {
            "n_examples": len(rows), "n_meta_ids": len(metas[s]),
            "label": counter(rows, lambda r: r["label"]),
            "category_id": counter(rows, lambda r: str(r["category_id"])),
            "group_x_label": counter(rows, lambda r: f"{r['source_group']}/{r['label']}"),
            "language": counter(rows, lambda r: r["language"]),
            "meta_ids_per_language": dict(sorted(collections.Counter(
                m.split("-")[2] for m in metas[s]).items())),
            "source_run": counter(rows, lambda r: r["source_run"]),
            "old_split": counter([e for e in ex if assign[e["meta_id"]] == s], lambda e: e["old_split"]),
            "prompt_chars_median": statistics.median(len(r["messages"][1]["content"]) for r in rows),
            "hardneg_type_by_label": counter(files["hard"][s], lambda r: f"{r['label']}:{r['neg_type']}"),
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out, stats)
    print(json.dumps({k: stats["splits"][k]["n_examples"] for k in SPLITS}), "->", out)


def write_readme(out: Path, st: dict) -> None:
    sp = st["splits"]
    L = ["# EU-Guard classifier data v1 (grouped re-split)", "",
         "Generated by `scripts/build_classifier_data.py`; do not edit by hand. Full numbers in `stats.json`.", "",
         f"Seed {st['seed']}; {st['n_total']} examples from 1,680 meta-prompts x 3 generation runs. "
         "Split 80/10/10 **by meta_id** (exact per-language meta counts, balanced over group x level cells).", "",
         "| split | examples | meta_ids | SAFE | EDGE_CASE | VIOLATION | qwen | gemini |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        d = sp[s]
        gem = sum(v for k, v in d["source_run"].items() if k.startswith("gemini"))
        L.append(f"| {s} | {d['n_examples']} | {d['n_meta_ids']} | " + " | ".join(
            str(d["label"].get(l, 0)) for l in LABELS) + f" | {d['source_run'].get('qwen_dataset_10k', 0)} | {gem} |")
    L += ["", "Leakage check (shared meta_ids / identical normalised prompts): "
          + ", ".join(f"{k}: {v['meta_ids']}/{v['prompts']}" for k, v in st["leakage_check"].items()), "",
          "## Files", "",
          "- `sft_*.jsonl` - `messages` [system, user, assistant] + metadata",
          "- `dpo_fixed_*.jsonl` - TRL conversational pairs, original rule-fixed rejected "
          "(SAFE -> VIOLATION/G1, otherwise -> SAFE)",
          "- `dpo_hardneg_*.jsonl` - same prompts/chosen, hard rejected; `neg_type` records which kind",
          "- `eval_{val,test}.jsonl` - `prompt` messages + `gold` JSON string, for generation-based scoring", "",
          "The jsonl files are gitignored (~170 MB); rebuild with `python scripts/build_classifier_data.py` "
          "(deterministic for a given seed).", "",
          "Metadata fields: example_id, concept_id (= meta_id), split, language, label, category_id, source_group, "
          "scenario_type, augmentation, source_run, generator, target (SAFE -> should_answer, else should_refuse; "
          "for compatibility with eu-guard-dpo-main's loader), pair_type (= gold label).", "",
          "## Hard-negative mix (train)", "", "| gold:neg_type | n |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in sp["train"]["hardneg_type_by_label"].items()]
    L += ["", "- level_swap: EDGE_CASE <-> VIOLATION, same category",
          "- wrong_category: same label, different G category",
          "- predict_safe: SAFE/null (the old fixed negative, kept at 20%)",
          "- flag_source_group: SAFE gold -> EDGE_CASE or VIOLATION with the meta-prompt's source group",
          "- flag_random_group: SAFE gold -> EDGE_CASE or VIOLATION with a random group", ""]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
