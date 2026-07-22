#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from sve import inject_sve


def read_jsonl(path, limit=None):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_images(paths):
    return [Image.open(p).convert('RGB') for p in paths]


def to_qwen_messages(messages):
    converted = []
    for msg in messages:
        content = msg['content']
        if isinstance(content, str) and '<image>' in content:
            parts = content.split('<image>')
            blocks = []
            for i, part in enumerate(parts):
                if i > 0:
                    blocks.append({'type': 'image'})
                if part:
                    blocks.append({'type': 'text', 'text': part.lstrip('\n')})
            converted.append({'role': msg['role'], 'content': blocks})
        else:
            converted.append(msg)
    return converted


def apply_chat_template_no_think(processor, messages, **kwargs):
    try:
        text = processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = processor.apply_chat_template(messages, **kwargs)
    if kwargs.get('add_generation_prompt') and isinstance(text, str):
        if text.endswith('<think>\n'):
            text = text[:-len('<think>\n')] + '<think>\n\n</think>\n\n'
        elif text.endswith('<think>'):
            text = text[:-len('<think>')] + '<think>\n\n</think>\n\n'
    return text


_BBOX_ARRAY_RE = re.compile(r'\[\s*\{.*?\}\s*\]', re.S)


def parse_boxes(text):
    """None = unparsable, [] = explicit no-defect, [...] = predicted/ground-truth boxes."""
    text = text.strip()
    if not text or 'no obvious defect' in text.lower():
        return []
    for candidate in (text, (_BBOX_ARRAY_RE.search(text) or [None])[0] if _BBOX_ARRAY_RE.search(text) else None):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [d['bbox_2d'] for d in data if isinstance(d, dict) and 'bbox_2d' in d]
        except Exception:
            continue
    return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return 0.0
    return max(iou(p, g) for p in pred_boxes for g in gt_boxes)


@torch.no_grad()
def generate_answer(model, processor, sample, device, max_pixels, max_new_tokens):
    messages = to_qwen_messages(sample['messages'])
    images = load_images(sample['images'])
    prompt_text = apply_chat_template_no_think(processor, messages[:-1], tokenize=False, add_generation_prompt=True)
    image_kwargs = {'images_kwargs': {'size': {'longest_edge': max_pixels, 'shortest_edge': 65536}}} if max_pixels else {}
    inputs = processor(text=[prompt_text], images=images, return_tensors='pt', **image_kwargs)
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    # model.generate() can't thread image_seqlens_per_sample through like a normal
    # forward call, so set the eval-time fallback the SVE hook reads instead.
    model.model.visual._eval_image_seqlens = [2]
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out_ids[0][inputs['input_ids'].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def summarize(results, iou_threshold, out_dir):
    positives = [r for r in results if r['has_defect']]
    negatives = [r for r in results if not r['has_defect']]

    def flagged(r):
        return r['pred_boxes'] is not None and len(r['pred_boxes']) > 0

    detected = [r for r in positives if flagged(r) and r['iou'] >= iou_threshold]
    false_positives = [r for r in negatives if flagged(r)]
    unparsable = [r for r in results if r['pred_boxes'] is None]

    summary = {
        'n_total': len(results),
        'n_positive': len(positives),
        'n_negative': len(negatives),
        'n_unparsable_predictions': len(unparsable),
        'iou_threshold': iou_threshold,
        'recall': len(detected) / len(positives) if positives else None,
        'miss_rate': 1 - len(detected) / len(positives) if positives else None,
        'over_detection_rate': len(false_positives) / len(negatives) if negatives else None,
        'mIoU_over_all_positives': sum(r['iou'] for r in positives) / len(positives) if positives else None,
        'mIoU_over_detected_only': sum(r['iou'] for r in detected) / len(detected) if detected else None,
        'n_detected': len(detected),
        'n_missed': len(positives) - len(detected),
        'n_false_positive': len(false_positives),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='/home/chang/models/Qwen3.5-2B')
    ap.add_argument('--checkpoint', required=True, help='trainable_state.pt from training')
    ap.add_argument('--test_jsonl', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--iou_threshold', type=float, default=0.5)
    ap.add_argument('--max_pixels', type=int, default=589824)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    inject_sve(model, proj_init_std=0.0)
    state = torch.load(args.checkpoint, map_location=device)
    _, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f'[load] checkpoint={args.checkpoint} loaded_tensors={len(state)} unexpected={len(unexpected)}', flush=True)
    model.eval()

    rows = read_jsonl(args.test_jsonl, args.limit or None)
    if rank == 0:
        print(f'[data] test={len(rows)}', flush=True)

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    results = []
    shard = list(range(rank, len(rows), world_size))
    for n, i in enumerate(shard):
        sample = rows[i]
        gt_boxes = parse_boxes(sample['messages'][-1]['content']) or []
        pred_text = generate_answer(model, processor, sample, device, args.max_pixels, args.max_new_tokens)
        pred_boxes = parse_boxes(pred_text)
        results.append({
            'sample_id': (sample.get('meta') or {}).get('sample_id'),
            'product': (sample.get('meta') or {}).get('product'),
            'has_defect': len(gt_boxes) > 0,
            'gt_boxes': gt_boxes,
            'pred_text': pred_text,
            'pred_boxes': pred_boxes,
            'iou': best_iou(pred_boxes, gt_boxes),
        })
        if rank == 0 and n % 20 == 0:
            print(f'[eval] {n}/{len(shard)} (this rank)', flush=True)

    rank_path = out_dir / f'preds_rank{rank}.jsonl'
    with rank_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dist.barrier()

    if rank == 0:
        all_results = []
        for p in sorted(out_dir.glob('preds_rank*.jsonl')):
            all_results.extend(read_jsonl(p))
        summarize(all_results, args.iou_threshold, out_dir)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
