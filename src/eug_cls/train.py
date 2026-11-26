"""Train the EU-Guard classifier with SFT (TRL SFTTrainer) or DPO (TRL DPOTrainer) + LoRA.

Mirrors eu-guard-dpo-main/src/eug_dpo/train.py (same config conventions, hyperparameters, length
filter, effective batch, adapter merge) but for the classifier data from
scripts/build_classifier_data.py, and with an SFT mode so that SFT and SFT->DPO runs share one code path.

Usage:
    python -m eug_cls.train --config configs/aya8b/dpo_hardneg.yaml
    python -m eug_cls.train --config configs/smoke.yaml --set method=sft train.max_steps=2
    accelerate launch --num_processes N -m eug_cls.train --config ...

Config sections:
  base     : optional parent YAML (relative path); this file is merged over it (recursively)
  method   : sft | dpo
  model    : name_or_path (HF id, or runs/<sft run>/merged for SFT->DPO), dtype, attn_implementation;
             load_in_4bit (QLoRA, NF4; no merged model is written), bnb_compute_dtype (default float16);
             init_adapter (dpo only): local dir or hub id of an adapter to continue training (SFT->DPO
             without merging); the frozen copy TRL makes of it is the DPO reference
  data     : train / validation files, drop_over_length, max_train_samples, max_eval_samples, languages
  lora     : peft.LoraConfig kwargs, or null for full fine-tuning
  train    : TrainingArguments kwargs shared by both methods (output_dir, lr, schedule, ...)
  sft      : extra trl.SFTConfig kwargs (used when method=sft)
  dpo      : extra trl.DPOConfig kwargs (used when method=dpo)
  effective_batch_size : sets gradient_accumulation_steps = this / (per-device batch x n GPUs)
  merge_adapter : also save a merged model to <output_dir>/merged (needed for vLLM eval and SFT->DPO)
  hub      : optional {repo_id | namespace, private}: upload the final adapter + run files (not merged/,
             checkpoints) to repo_id, or to <namespace>/eug-cls-<run_name>
  wandb_project / wandb_entity : used when train.report_to=wandb

SFT is trained on the assistant completion only (prompt-completion format -> TRL completion_only_loss).
DPO with LoRA uses the base model with the adapter disabled as the reference model; for SFT->DPO the base
is the merged SFT model, so the reference is the SFT policy.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
from pathlib import Path

import yaml

SPLITS_FORBIDDEN = ("test",)


# ---------------------------------------------------------------- config
def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if base := cfg.pop("base", None):
        cfg = merge(load_config(str(Path(path).parent / base)), cfg)
    return load_config_overrides(cfg, overrides)


def load_config_overrides(cfg: dict, overrides) -> dict:
    """Apply `a.b.c=value` overrides (values parsed as YAML) in place; returns cfg."""
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[leaf] = parse_value(val)
    return cfg


def resolve_grad_accum(cfg: dict, world_size: int) -> None:
    eff = cfg.get("effective_batch_size")
    if not eff:
        return
    per_step = cfg["train"].get("per_device_train_batch_size", 8) * world_size
    if eff % per_step:
        raise ValueError(f"effective_batch_size {eff} not divisible by per-device batch x GPUs = {per_step}")
    cfg["train"]["gradient_accumulation_steps"] = eff // per_step


# ---------------------------------------------------------------- data
def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path: str, method: str) -> list[dict]:
    """Rows as {prompt, completion} (sft) or {prompt, chosen, rejected} (dpo) message lists + metadata."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: empty")
    out = []
    for i, r in enumerate(rows, 1):
        if r.get("split") in SPLITS_FORBIDDEN:
            raise ValueError(f"{path}:{i}: row from split {r['split']!r} -- refusing to train on it")
        if method == "sft":
            if "messages" not in r or r["messages"][-1]["role"] != "assistant":
                raise ValueError(f"{path}:{i}: SFT row needs `messages` ending in an assistant turn")
            r = {**r, "prompt": r["messages"][:-1], "completion": r["messages"][-1:]}
        else:
            for k in ("prompt", "chosen", "rejected"):
                if not isinstance(r.get(k), list):
                    raise ValueError(f"{path}:{i}: DPO row needs `{k}` as a message list")
        if r["prompt"][-1]["role"] != "user":
            raise ValueError(f"{path}:{i}: prompt must end in a user turn")
        out.append(r)
    return out


def select(rows, languages, max_n, seed):
    if languages:
        rows = [r for r in rows if r["language"] in set(languages)]
    if max_n is not None and len(rows) > max_n:
        rows = random.Random(seed).sample(rows, max_n)
    return rows


def completions(r: dict, method: str) -> list[list[dict]]:
    return [r["completion"]] if method == "sft" else [r["chosen"], r["rejected"]]


def drop_over_length(rows, tok, max_length: int, method: str):
    def n_tok(prompt, comp):
        text = tok.apply_chat_template(prompt + comp, tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    kept, dropped, lens = [], collections.Counter(), []
    for r in rows:
        n = max(n_tok(r["prompt"], c) for c in completions(r, method))
        lens.append(n)
        if n <= max_length:
            kept.append(r)
        else:
            dropped[r["language"]] += 1
    lens.sort()
    return kept, {"max_length": max_length, "n_in": len(rows), "n_dropped": len(rows) - len(kept),
                  "dropped_by_language": dict(sorted(dropped.items())),
                  "tokens_p50": lens[len(lens) // 2], "tokens_p99": lens[int(len(lens) * 0.99)],
                  "tokens_max": lens[-1]}


def row_stats(rows: list[dict], method: str) -> dict:
    s = {"n": len(rows),
         "by_label": dict(sorted(collections.Counter(r.get("label") for r in rows).items())),
         "by_language": dict(sorted(collections.Counter(r.get("language") for r in rows).items()))}
    if method == "dpo":
        s["by_neg_type"] = dict(sorted(collections.Counter(r.get("neg_type") for r in rows).items()))
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overlay", action="append", default=[], metavar="YAML",
                    help="merged over --config before --set, e.g. configs/kaggle.yaml (repeatable)")
    ap.add_argument("--resume", metavar="last|PATH")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for ov in a.overlay:
        cfg = merge(cfg, load_config(ov))
    cfg = load_config_overrides(cfg, a.set)
    method = cfg.get("method")
    if method not in ("sft", "dpo"):
        raise ValueError(f"method must be sft or dpo, got {method!r}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolve_grad_accum(cfg, world_size)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    mcfg, dcfg = cfg["model"], cfg["data"]
    targs = {**cfg["train"], **(cfg.get(method) or {})}
    targs.setdefault("run_name", Path(targs["output_dir"]).name)
    for k in ("wandb_project", "wandb_entity"):
        if cfg.get(k):
            os.environ.setdefault(k.upper(), cfg[k])
    os.environ.setdefault("WANDB_DIR", targs["output_dir"])
    seed = targs.get("seed", 42)

    resume = a.resume
    if resume == "last":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(targs["output_dir"])
        if resume is None:
            raise FileNotFoundError(f"--resume last: no checkpoint-* in {targs['output_dir']}")

    tok = AutoTokenizer.from_pretrained(mcfg["name_or_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    langs = dcfg.get("languages")
    train_rows = select(load_rows(dcfg["train"], method), langs, None, seed)
    eval_rows = select(load_rows(dcfg["validation"], method), langs, None, seed) if dcfg.get("validation") else None
    length_filter = None
    if dcfg.get("drop_over_length"):
        max_len = targs.get("max_length")
        if not max_len:
            raise ValueError("data.drop_over_length needs an explicit max_length in the sft/dpo section")
        # subsample first here (unlike the reference) so tiny smoke runs don't tokenise 22k rows
        train_rows = select(train_rows, None, dcfg.get("max_train_samples"), seed)
        train_rows, lf_train = drop_over_length(train_rows, tok, max_len, method)
        length_filter = {"train": lf_train}
        if eval_rows:
            eval_rows = select(eval_rows, None, dcfg.get("max_eval_samples"), seed)
            eval_rows, length_filter["validation"] = drop_over_length(eval_rows, tok, max_len, method)
        print(json.dumps({"length_filter": length_filter}, indent=2))
    train_rows = select(train_rows, None, dcfg.get("max_train_samples"), seed)
    if eval_rows:
        eval_rows = select(eval_rows, None, dcfg.get("max_eval_samples"), seed)

    keep = ("prompt", "completion") if method == "sft" else ("prompt", "chosen", "rejected")

    def to_ds(rows):
        return Dataset.from_list([{k: r[k] for k in keep} for r in rows])

    dtype = mcfg.get("dtype")
    own_keys = ("name_or_path", "load_in_4bit", "bnb_compute_dtype", "init_adapter")
    model_init_kwargs = {k: v for k, v in mcfg.items() if k not in own_keys and v is not None}
    if isinstance(dtype, str) and dtype != "auto":
        model_init_kwargs["dtype"] = getattr(torch, dtype)
    quantized = bool(mcfg.get("load_in_4bit"))
    if quantized:   # QLoRA (e.g. Kaggle T4s): NF4 base, LoRA adapters trained on top
        from transformers import BitsAndBytesConfig
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, mcfg.get("bnb_compute_dtype", "float16")))
        model_init_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if not dcfg.get("validation"):
        targs["eval_strategy"] = "no"
    peft_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]) if cfg.get("lora") else None

    model = mcfg["name_or_path"]
    if mcfg.get("init_adapter"):
        # Continue training an existing adapter (SFT->DPO without a merged model, e.g. on QLoRA). TRL's
        # DPOTrainer copies the loaded adapter into a frozen "ref" adapter, so the reference is the SFT policy.
        if method != "dpo":
            raise ValueError("model.init_adapter is only supported for method=dpo")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(mcfg["name_or_path"], **model_init_kwargs)
        model = PeftModel.from_pretrained(base, mcfg["init_adapter"], is_trainable=True)
        peft_config = None
    init_kw = {"model_init_kwargs": model_init_kwargs} if isinstance(model, str) else {}

    if method == "sft":
        from trl import SFTConfig, SFTTrainer
        targs.setdefault("completion_only_loss", True)
        args = SFTConfig(**targs, **init_kw)
        Trainer = SFTTrainer
    else:
        from trl import DPOConfig, DPOTrainer
        args = DPOConfig(**targs, **init_kw)
        Trainer = DPOTrainer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "method": method, "config": cfg, "world_size": world_size,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size,
        "data": {"train_file": dcfg["train"], "train_sha256": sha256(dcfg["train"]),
                 "validation_file": dcfg.get("validation"),
                 "validation_sha256": sha256(dcfg["validation"]) if dcfg.get("validation") else None},
        "length_filter": length_filter,
        "train_rows": row_stats(train_rows, method),
        "eval_rows": row_stats(eval_rows, method) if eval_rows else None,
        "resumed_from": resume,
    }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_ds(train_rows),
        eval_dataset=to_ds(eval_rows) if eval_rows else None,
        processing_class=tok,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process:
        (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: info[k] for k in ("method", "effective_batch_size", "train_rows")}, indent=2))

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    if eval_rows:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_model(str(out))
    if trainer.accelerator.is_main_process:
        tok.save_pretrained(str(out))

    hub_cfg = cfg.get("hub") or {}
    repo_id = hub_cfg.get("repo_id") or (f"{hub_cfg['namespace']}/eug-cls-{targs['run_name']}"
                                         if hub_cfg.get("namespace") else None)
    if repo_id and trainer.accelerator.is_main_process:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, private=hub_cfg.get("private", True), exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(out),
                          ignore_patterns=["checkpoint-*", "checkpoint-*/*", "merged/*", "wandb/*"])
        print(f"uploaded to {repo_id}")

    if cfg.get("merge_adapter") and peft_config is not None and not quantized:
        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process:
            merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
            merged.save_pretrained(str(out / "merged"))
            tok.save_pretrained(str(out / "merged"))
            print(f"merged model saved to {out / 'merged'}")


if __name__ == "__main__":
    main()
