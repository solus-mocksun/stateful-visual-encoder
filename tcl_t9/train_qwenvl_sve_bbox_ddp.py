#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

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

def build_inputs(processor, sample, device, max_pixels=None):
    messages = to_qwen_messages(sample['messages'])
    images = load_images(sample['images'])
    full_text = apply_chat_template_no_think(processor, messages, tokenize=False, add_generation_prompt=False)
    prompt_text = apply_chat_template_no_think(processor, messages[:-1], tokenize=False, add_generation_prompt=True)
    image_kwargs = {'images_kwargs': {'size': {'longest_edge': max_pixels, 'shortest_edge': 65536}}} if max_pixels else {}
    full = processor(text=[full_text], images=images, return_tensors='pt', **image_kwargs)
    prompt = processor(text=[prompt_text], images=images, return_tensors='pt', **image_kwargs)
    labels = full['input_ids'].clone()
    prompt_len = prompt['input_ids'].shape[1]
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    full['labels'] = labels
    return {k: (v.to(device, non_blocking=True) if hasattr(v, 'to') else v) for k, v in full.items()}

def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_names = []
    if mode in ('sve_only', 'sve_lm_head'):
        for name, module in model.named_modules():
            if type(module).__name__ == 'CrossFFNBlock':
                for p_name, p in module.named_parameters(recurse=True):
                    p.requires_grad_(True)
                    trainable_names.append(f'{name}.{p_name}')
    if mode == 'sve_lm_head':
        for name, p in model.named_parameters():
            if 'lm_head' in name.lower():
                p.requires_grad_(True)
                trainable_names.append(name)
    if mode == 'all':
        for name, p in model.named_parameters():
            p.requires_grad_(True)
            trainable_names.append(name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total, trainable_names[:20]

def save_trainable(model, out_dir, step, extra, rank):
    ckpt = Path(out_dir) / f'checkpoint-step-{step}'
    if rank == 0:
        ckpt.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items() if 'sve' in k.lower() or 'cross' in k.lower()}
        if not state:
            trainable = {name for name, p in model.named_parameters() if p.requires_grad}
            state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}
        torch.save(state, ckpt / 'trainable_state.pt')
        (ckpt / 'trainer_state.json').write_text(json.dumps(extra, indent=2), encoding='utf-8')
        print(f'[save] {ckpt}', flush=True)
    dist.barrier()
    return ckpt

@torch.no_grad()
def distributed_eval(model, processor, val_rows, device, rank, world_size, max_eval_samples):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)
    rows = val_rows[:max_eval_samples]
    for i in range(rank, len(rows), world_size):
        batch = build_inputs(processor, rows[i], device, getattr(model, '_sve_max_pixels', None))
        out = model(**batch, image_seqlens_per_sample=[2])
        total_loss += out.loss.detach().float()
        total_count += 1
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_count.clamp_min(1)).cpu())

def init_dist():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device('cuda', local_rank)

def rank0_print(rank, *args):
    if rank == 0:
        print(*args, flush=True)

def average_grads(params, world_size):
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='/home/chang/models/Qwen3.5-2B')
    ap.add_argument('--train_jsonl', required=True)
    ap.add_argument('--val_jsonl', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--max_train_samples', type=int, default=0)
    ap.add_argument('--max_eval_samples', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_mode', choices=['sve_only', 'sve_lm_head', 'all'], default='sve_only')
    ap.add_argument('--proj_init_std', type=float, default=0.0)
    ap.add_argument('--max_pixels', type=int, default=0)
    args = ap.parse_args()

    rank, world_size, local_rank, device = init_dist()
    random.seed(args.seed)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')

    rank0_print(rank, f'[dist] world_size={world_size} mode={args.train_mode}')
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    rank0_print(rank, '[load] model')
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    n_sve = inject_sve(model, proj_init_std=args.proj_init_std)
    rank0_print(rank, f'[sve] injected params={n_sve:,}')
    try:
        model.gradient_checkpointing_enable()
        rank0_print(rank, '[memory] gradient checkpointing enabled')
    except Exception as e:
        rank0_print(rank, f'[memory] gradient checkpointing unavailable: {e!r}')

    n_trainable, n_total, sample_names = set_trainable(model, args.train_mode)
    rank0_print(rank, f'[trainable] trainable={n_trainable:,} total={n_total:,} ratio={n_trainable/n_total:.4%}')
    rank0_print(rank, f'[trainable] examples={sample_names}')

    train_rows = read_jsonl(args.train_jsonl, args.max_train_samples or None)
    val_rows = read_jsonl(args.val_jsonl)
    random.Random(args.seed).shuffle(train_rows)
    rank0_print(rank, f'[data] train={len(train_rows)} val={len(val_rows)}')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.max_steps * args.warmup_ratio), args.max_steps)
    model._sve_max_pixels = args.max_pixels or None
    rank0_print(rank, f'[data] max_pixels={args.max_pixels or None}')
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    micro = 0
    global_micro = 0
    local_losses = []
    eval_history = []
    start = time.time()

    while step < args.max_steps:
        for _ in range(len(train_rows) // max(1, world_size)):
            sample_idx = (global_micro * world_size + rank) % len(train_rows)
            sample = train_rows[sample_idx]
            global_micro += 1
            micro += 1
            batch = build_inputs(processor, sample, device, args.max_pixels or None)
            out = model(**batch, image_seqlens_per_sample=[2])
            loss = out.loss / args.grad_accum
            loss.backward()
            local_losses.append(float(out.loss.detach().cpu()))

            if micro % args.grad_accum == 0:
                average_grads(params, world_size)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    recent = sum(local_losses[-args.log_every:]) / min(len(local_losses), args.log_every)
                    loss_tensor = torch.tensor([recent], device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    rank0_print(rank, f'[step {step:05d}] loss={loss_tensor.item():.4f} lr={scheduler.get_last_lr()[0]:.3e} elapsed_min={(time.time()-start)/60:.1f}')
                if args.eval_every and step % args.eval_every == 0:
                    val_loss = distributed_eval(model, processor, val_rows, device, rank, world_size, args.max_eval_samples)
                    if rank == 0:
                        eval_history.append({'step': step, 'val_loss': val_loss})
                    rank0_print(rank, f'[eval step {step:05d}] val_loss={val_loss:.4f}')
                if args.save_every and step % args.save_every == 0:
                    save_trainable(model, args.output_dir, step, {'step': step, 'eval_history': eval_history}, rank)
                if step >= args.max_steps:
                    break
        random.Random(args.seed + step).shuffle(train_rows)

    final_val = distributed_eval(model, processor, val_rows, device, rank, world_size, args.max_eval_samples)
    if rank == 0:
        eval_history.append({'step': step, 'val_loss': final_val, 'final': True})
    save_trainable(model, args.output_dir, step, {'step': step, 'eval_history': eval_history, 'final_val_loss': final_val, 'done': True}, rank)
    rank0_print(rank, f'[done] step={step} final_val_loss={final_val:.4f}')
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
