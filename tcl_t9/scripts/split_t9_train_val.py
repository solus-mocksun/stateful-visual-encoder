#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def key_for(row):
    meta = row.get("meta") or {}
    return (meta.get("product") or "UNKNOWN", bool(meta.get("has_defect")))

def summarize(rows):
    by_kind = Counter("with_defects" if (r.get("meta") or {}).get("has_defect") else "no_defects" for r in rows)
    by_product = Counter((r.get("meta") or {}).get("product") or "UNKNOWN" for r in rows)
    by_product_kind = Counter()
    for r in rows:
        meta = r.get("meta") or {}
        product = meta.get("product") or "UNKNOWN"
        kind = "with_defects" if meta.get("has_defect") else "no_defects"
        by_product_kind[f"{product}/{kind}"] += 1
    return {
        "total": len(rows),
        "by_kind": dict(by_kind),
        "by_product": dict(by_product),
        "by_product_kind": dict(sorted(by_product_kind.items())),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tcl_t9/manifests/t9_bbox_sft_train_full.jsonl")
    ap.add_argument("--out_dir", default="tcl_t9/manifests")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[key_for(row)].append(row)

    rng = random.Random(args.seed)
    train, val = [], []
    group_summary = {}
    for key, items in sorted(groups.items()):
        rng.shuffle(items)
        n_val = round(len(items) * args.val_ratio)
        if len(items) > 1:
            n_val = max(1, min(len(items) - 1, n_val))
        val_items = items[:n_val]
        train_items = items[n_val:]
        product, has_defect = key
        group_summary[f"{product}/{'with_defects' if has_defect else 'no_defects'}"] = {
            "total": len(items),
            "train": len(train_items),
            "val": len(val_items),
        }
        train.extend(train_items)
        val.extend(val_items)

    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = Path(args.out_dir)
    train_path = out_dir / "t9_bbox_sft_train_stratified85.jsonl"
    val_path = out_dir / "t9_bbox_sft_val_stratified15.jsonl"
    summary_path = out_dir / "t9_bbox_sft_stratified_split_summary.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    train_ids = {(r.get("meta") or {}).get("sample_id") for r in train}
    val_ids = {(r.get("meta") or {}).get("sample_id") for r in val}
    overlap = sorted(x for x in train_ids.intersection(val_ids) if x)

    summary = {
        "input": args.input,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train": str(train_path),
        "val": str(val_path),
        "train_summary": summarize(train),
        "val_summary": summarize(val),
        "group_split": group_summary,
        "overlap_sample_id_count": len(overlap),
        "overlap_sample_id_examples": overlap[:20],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
