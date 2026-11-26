"""Zip what Kaggle needs (code + raw inputs of build_classifier_data.py) for upload as a Kaggle Dataset.

    python scripts/make_kaggle_bundle.py            # -> kaggle_bundle/eu-guard-cls.zip

Upload: kaggle.com -> Datasets -> New Dataset -> drop the zip (Kaggle unpacks it). Re-upload a new
version after code changes. data/classifier_v1 is rebuilt on Kaggle (deterministic), not uploaded.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "pyproject.toml", "CLAUDE.md",
    "src/eug_cls/*.py", "scripts/*.py", "configs/*.yaml", "configs/*.csv", "configs/*/*.yaml",
    "kaggle/*.ipynb",
    "merged_datasets/dpo_*_combined.jsonl", "merged_datasets/qwen_train_combined.jsonl",  # system prompt source
    "qwen_dataset_10k/metadata.jsonl", "gemini_dataset_10k/metadata.jsonl", "gemini_dataset_20k/metadata.jsonl",
]


def main() -> None:
    out = ROOT / "kaggle_bundle" / "eu-guard-cls.zip"
    out.parent.mkdir(exist_ok=True)
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file()})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{len(files)} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
