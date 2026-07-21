#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

SYSTEM = "You are an industrial panel defect detection assistant. Compare a normal template image with a test image and locate defects in the test image."
USER1 = "<image>\nThis is the normal template image for this panel region."
FILLER = "I have seen the normal template image. Please provide the test image."
USER2 = 'This is the test image. Compare it with the template image. If there is a defect, output a strict JSON array with items in the format {"bbox_2d": [x1, y1, x2, y2], "label": "defect"}. Coordinates must be normalized integers from 0 to 1000. If there is no obvious defect, output exactly: no obvious defect.'
USER2 = "<image>\n" + USER2
NO_DEFECT = "no obvious defect"

IMAGE_EXTS = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
TEST_IMAGE_EXTS = [".jpg", ".jpeg", ".JPG", ".JPEG"]

def norm_bbox(box, w, h):
    x1, y1, x2, y2 = box
    return [
        max(0, min(1000, round(x1 / w * 1000))),
        max(0, min(1000, round(y1 / h * 1000))),
        max(0, min(1000, round(x2 / w * 1000))),
        max(0, min(1000, round(y2 / h * 1000))),
    ]

def bbox_from_labelme(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, None, f"bad_json:{type(e).__name__}"
    w, h = obj.get("imageWidth"), obj.get("imageHeight")
    if not w or not h:
        return None, None, "missing_image_size"
    xs, ys = [], []
    for sh in obj.get("shapes") or []:
        for p in sh.get("points") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if not xs or not ys:
        return None, (w, h), "empty_shapes"
    x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None, (w, h), "degenerate_bbox"
    return norm_bbox((x1, y1, x2, y2), w, h), (w, h), ""

def image_size_or_error(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size, ""
    except Exception as e:
        return None, f"bad_image:{type(e).__name__}"

def find_test_image(folder, stem):
    for ext in TEST_IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists() and not p.name.endswith("_template.jpg"):
            return p
    return None

def find_template(folder, stem):
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        p = folder / f"{stem}_template{ext}"
        if p.exists():
            return p
    return None

def no_defect_images(folder):
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith("_template.jpg") or p.name.endswith("_template.jpeg"):
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        yield p

def build_sample(product, sample_id, image, template, has_defect, target, label_json=None, mask=None):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER1},
            {"role": "assistant", "content": FILLER},
            {"role": "user", "content": USER2},
            {"role": "assistant", "content": target},
        ],
        "images": [str(template), str(image)],
        "meta": {
            "sample_id": sample_id,
            "product": product,
            "has_defect": has_defect,
            "mask": str(mask) if mask else None,
            "label_json": str(label_json) if label_json else None,
            "source": "adc_Dataset/T9/train",
        },
    }

def add_skip(skipped, product, kind, stem, reason, path):
    skipped.append({
        "product": product,
        "kind": kind,
        "sample_id": stem,
        "reason": reason,
        "path": str(path) if path else None,
    })

def collect(root, verify_images=True):
    rows = []
    skipped = []
    stats = Counter()
    by_product = Counter()
    by_kind = Counter()

    for prod_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        product = prod_dir.name
        defect_dir = prod_dir / "with_defects"
        if defect_dir.exists():
            for jp in sorted(defect_dir.glob("*.json")):
                stem = jp.stem
                image = find_test_image(defect_dir, stem)
                template = find_template(defect_dir, stem)
                mask = defect_dir / f"{stem}.png"
                if image is None:
                    stats["defect_missing_image"] += 1
                    add_skip(skipped, product, "with_defects", stem, "missing_test_image", jp)
                    continue
                if template is None:
                    stats["defect_missing_template"] += 1
                    add_skip(skipped, product, "with_defects", stem, "missing_template", image)
                    continue
                bbox, size, err = bbox_from_labelme(jp)
                if bbox is None:
                    stats[f"defect_bad_label:{err}"] += 1
                    add_skip(skipped, product, "with_defects", stem, err, jp)
                    continue
                for role, p in [("test_image", image), ("template", template)]:
                    if verify_images:
                        _, img_err = image_size_or_error(p)
                        if img_err:
                            stats[f"defect_{role}_{img_err}"] += 1
                            add_skip(skipped, product, "with_defects", stem, f"{role}_{img_err}", p)
                            break
                else:
                    target = json.dumps([{"bbox_2d": bbox, "label": "defect"}], ensure_ascii=False)
                    rows.append(build_sample(product, stem, image, template, True, target, jp, mask if mask.exists() else None))
                    by_product[product] += 1
                    by_kind["with_defects"] += 1

        normal_dir = prod_dir / "no_defects"
        if normal_dir.exists():
            for image in no_defect_images(normal_dir):
                stem = image.stem
                template = find_template(normal_dir, stem)
                if template is None:
                    stats["normal_missing_template"] += 1
                    add_skip(skipped, product, "no_defects", stem, "missing_template", image)
                    continue
                for role, p in [("test_image", image), ("template", template)]:
                    if verify_images:
                        _, img_err = image_size_or_error(p)
                        if img_err:
                            stats[f"normal_{role}_{img_err}"] += 1
                            add_skip(skipped, product, "no_defects", stem, f"{role}_{img_err}", p)
                            break
                else:
                    rows.append(build_sample(product, stem, image, template, False, NO_DEFECT))
                    by_product[product] += 1
                    by_kind["no_defects"] += 1

    return rows, skipped, stats, by_product, by_kind

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data2/shared/TCL-U/adc&adr/pss/data/adc_Dataset/T9/train")
    ap.add_argument("--out_dir", default="tcl_t9/manifests")
    ap.add_argument("--val_samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_verify_images", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    rows, skipped, stats, by_product, by_kind = collect(root, verify_images=not args.no_verify_images)
    random.Random(args.seed).shuffle(rows)
    val = rows[: min(args.val_samples, len(rows))]

    train_path = out_dir / "t9_bbox_sft_train_full.jsonl"
    val_path = out_dir / "t9_bbox_sft_val_sample.jsonl"
    skipped_path = out_dir / "t9_unusable_samples.jsonl"
    summary_path = out_dir / "t9_bbox_sft_summary.json"

    write_jsonl(train_path, rows)
    write_jsonl(val_path, val)
    write_jsonl(skipped_path, skipped)

    summary = {
        "root": str(root),
        "train_full": str(train_path),
        "val_sample": str(val_path),
        "unusable_samples": str(skipped_path),
        "total_usable": len(rows),
        "val_sample_count": len(val),
        "by_kind": dict(by_kind),
        "by_product": dict(by_product),
        "skipped_count": len(skipped),
        "skipped_by_reason": dict(stats),
        "note": "val_sample is sampled from full train and not removed from train_full.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
