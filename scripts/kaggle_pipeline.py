"""Run the whole experiment for one base model on Kaggle (2x T4), resumable across sessions.

Stages (in order): fewshot, sft, dpo_fixed, dpo_hardneg, sftdpo_fixed, sftdpo_hardneg, score.
Each training stage trains with QLoRA (configs/kaggle.yaml overlay), uploads the adapter to the private
HF repo <namespace>/eug-cls-<model>_<stage>, predicts the test split with it and uploads the predictions
to the same repo (eval/test.jsonl). A stage whose predictions are already on the hub is skipped, so
re-running `--stage all` in a new session continues where the last one stopped. `--hours` stops before
starting a stage that would likely not finish in the session (estimated from the previous stage).

    python scripts/kaggle_pipeline.py --model aya8b --stage pilot          # ~30 min: measures s/step
    python scripts/kaggle_pipeline.py --model aya8b --stage all --hours 11
    python scripts/kaggle_pipeline.py --model aya8b --stage score          # report from what is on the hub

Needs HF_TOKEN (write access; aya / Llama licenses accepted) and --hf-namespace (or HF_NAMESPACE).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_STAGES = ["sft", "dpo_fixed", "dpo_hardneg", "sftdpo_fixed", "sftdpo_hardneg"]
ALL = ["fewshot"] + TRAIN_STAGES
DELTAS = ["sft:fewshot", "dpo_fixed:fewshot", "dpo_hardneg:fewshot", "dpo_hardneg:dpo_fixed",
          "sftdpo_fixed:sft", "sftdpo_hardneg:sft", "sftdpo_hardneg:sftdpo_fixed"]
FEW_SHOT = 6


def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def base_model(model: str) -> str:
    cfg = yaml.safe_load((ROOT / "configs" / model / "base.yaml").read_text(encoding="utf-8"))
    return cfg["model"]["name_or_path"]


def repo(ns: str, model: str, stage: str) -> str:
    return f"{ns}/eug-cls-{model}_{stage}"


def done(api, repo_id: str) -> bool:
    try:
        return api.file_exists(repo_id, "eval/test.jsonl")
    except Exception:
        return False


def ensure_data() -> None:
    if not (ROOT / "data/classifier_v1/eval_test.jsonl").exists():
        sh([sys.executable, "scripts/build_classifier_data.py"])


def n_gpus() -> int:
    import torch
    return torch.cuda.device_count()


def train(model: str, stage: str, ns: str | None, extra: list[str]) -> Path:
    out = f"runs/{model}_{stage}"
    sets = [f"train.output_dir={out}", f"hub.namespace={ns or 'null'}", *extra]
    if stage.startswith("sftdpo"):
        # continue the SFT adapter (from the hub) instead of training on merged SFT weights
        sets += [f"model.name_or_path={base_model(model)}", f"model.init_adapter={repo(ns, model, 'sft')}"]
    cmd = ["-m", "eug_cls.train", "--config", f"configs/{model}/{stage}.yaml",
           "--overlay", "configs/kaggle.yaml", "--set", *sets]
    g = n_gpus()
    if g > 1:
        sh([sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(g),
            "--mixed_precision", "fp16", *cmd])
    else:
        sh([sys.executable, *cmd])
    return ROOT / out


def predict(model: str, out: Path, adapter: str | None, few_shot: int, limit: int | None, bs: int) -> None:
    cmd = [sys.executable, "-m", "eug_cls.predict", "--backend", "hf", "--load-in-4bit", "--dtype", "float16",
           "--model", base_model(model), "--batch-size", str(bs), "--out", str(out)]
    if adapter:
        cmd += ["--adapter", adapter]
    if few_shot:
        cmd += ["--few-shot", str(few_shot)]
    if limit:
        cmd += ["--limit", str(limit)]
    sh(cmd)


def upload_eval(api, repo_id: str, pred: Path) -> None:
    api.create_repo(repo_id, private=True, exist_ok=True)
    for p in (pred, Path(str(pred) + ".meta.json")):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"eval/{p.name}", repo_id=repo_id)
    print(f"uploaded predictions to {repo_id}/eval/", flush=True)


def run_stage(api, a, stage: str) -> None:
    pred = ROOT / f"runs/eval/{a.model}_{stage}/test.jsonl"
    if stage == "fewshot":
        predict(a.model, pred, None, FEW_SHOT, None, a.batch_size)
    else:
        out = train(a.model, stage, a.hf_namespace, [f"data.max_train_samples={a.max_train_samples}"])
        predict(a.model, pred, str(out), 0, None, a.batch_size)
    upload_eval(api, repo(a.hf_namespace, a.model, stage), pred)


def score(api, a) -> None:
    from huggingface_hub import hf_hub_download
    runs = []
    for stage in ALL:
        rid = repo(a.hf_namespace, a.model, stage)
        if done(api, rid):
            p = hf_hub_download(rid, "eval/test.jsonl", local_dir=ROOT / f"runs/eval_hub/{a.model}_{stage}")
            runs.append(f"{stage}={p}")
        else:
            print(f"  (no predictions yet for {stage})")
    names = {r.split("=")[0] for r in runs}
    if len(names) < 1:
        raise SystemExit("nothing to score yet")
    deltas = [d for d in DELTAS if set(d.split(":")) <= names]
    out = ROOT / f"results/{a.model}_test"
    cmd = [sys.executable, "-m", "eug_cls.score", "--out", str(out), "--by", "generator", "--by", "source_group"]
    for r in runs:
        cmd += ["--run", r]
    for d in deltas:
        cmd += ["--delta", d]
    sh(cmd)
    rid = f"{a.hf_namespace}/eug-cls-results"
    api.create_repo(rid, private=True, exist_ok=True)
    for suf in (".md", ".json", "_per_language.csv"):
        api.upload_file(path_or_fileobj=f"{out}{suf}", path_in_repo=f"{a.model}_test{suf}", repo_id=rid)
    print(f"report: {out}.md (also uploaded to {rid})")


def pilot(a) -> None:
    """Short DPO run + small eval: checks memory and the code path, and measures s/step."""
    t0 = time.time()
    out = train(a.model, "dpo_hardneg", None, ["train.max_steps=10", "train.eval_steps=10",
                                               "data.max_train_samples=200", "data.max_eval_samples=16",
                                               f"train.output_dir=runs/pilot_{a.model}", "train.save_strategy=no"])
    t_train = time.time() - t0
    out = ROOT / f"runs/pilot_{a.model}"
    state = json.loads((out / "trainer_state.json").read_text())
    rt = next((h["train_runtime"] for h in state["log_history"] if "train_runtime" in h), t_train)
    t1 = time.time()
    predict(a.model, ROOT / f"runs/pilot_{a.model}/eval.jsonl", str(out), 0, 64, a.batch_size)
    t_pred = time.time() - t1
    s_per_step = rt / 10
    g = max(n_gpus(), 1)
    s_per_pair = s_per_step / 16      # effective batch 16 pairs per optimizer step (configs/kaggle.yaml)
    print(json.dumps({
        "gpus": g, "dpo_seconds_per_optimizer_step(eff_batch_16)": round(s_per_step, 1),
        "dpo_hours_per_1000_pairs": round(s_per_pair * 1000 / 3600, 2),
        "predict_seconds_per_64": round(t_pred, 1),
        "predict_minutes_full_test(2839)": round(t_pred / 64 * 2839 / 60, 1),
        "hint": "pick --max-train-samples so one DPO stage takes <= ~9 h (SFT is ~2-3x faster per example)",
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["aya8b", "llama8b"])
    ap.add_argument("--stage", required=True, choices=["pilot", "all", "score", *ALL])
    ap.add_argument("--hf-namespace", default=os.environ.get("HF_NAMESPACE"))
    ap.add_argument("--max-train-samples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16, help="generation batch size")
    ap.add_argument("--hours", type=float, default=11.0, help="session budget for --stage all")
    a = ap.parse_args()
    ensure_data()
    if a.stage == "pilot":
        return pilot(a)
    if not a.hf_namespace:
        raise SystemExit("set --hf-namespace or HF_NAMESPACE (your HF username): adapters live there")
    from huggingface_hub import HfApi
    api = HfApi()
    if a.stage == "score":
        return score(api, a)
    stages = ALL if a.stage == "all" else [a.stage]
    t_start, last = time.time(), 0.0
    for st in stages:
        rid = repo(a.hf_namespace, a.model, st)
        if done(api, rid):
            print(f"[skip] {st}: predictions already on {rid}")
            continue
        if st.startswith("sftdpo") and not done(api, repo(a.hf_namespace, a.model, "sft")):
            raise SystemExit("sftdpo needs the sft stage first")
        elapsed = (time.time() - t_start) / 3600
        if a.stage == "all" and last and elapsed + last * 1.1 > a.hours:
            print(f"[stop] {elapsed:.1f} h used; next stage (~{last:.1f} h) may not fit. Re-run in a new session.")
            break
        t0 = time.time()
        run_stage(api, a, st)
        dur = (time.time() - t0) / 3600
        last = max(last, dur) if st != "fewshot" else last
        print(f"[done] {st} in {dur:.2f} h", flush=True)
    if a.stage == "all":
        score(api, a)


if __name__ == "__main__":
    main()
