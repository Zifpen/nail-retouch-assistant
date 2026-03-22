# Nail Retouch Assistant

This repository prepares paired manicure retouch data for LoRA training and future inference services.

## Raw dataset

Expected raw layout:

```text
raw/
  pair_0001/
    before.jpg
    after.jpg
```

## Preprocess dataset

Install dependencies:

```bash
python3 -m pip install pillow
```

Build processed train/val splits:

```bash
python3 src/data/preprocess_pairs.py --raw-dir raw --output-dir dataset/processed --size 768
```

Apply structured tags to metadata:

```bash
python3 src/data/apply_annotations.py
```

Outputs:

```text
dataset/processed/
  train/
    images/
    metadata.jsonl
  val/
    images/
    metadata.jsonl
```

Each metadata row contains:

- paired source image path
- paired target image path
- default prompt
- default negative prompt
- optional structured tags: `shape`, `finish`, `color_family`
