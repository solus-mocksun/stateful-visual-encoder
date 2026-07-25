#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from sve import inject_sve


def read_jsonl(path, limit=None):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows)>=limit:
                    break
    return rows


def to_qwen_messages(messages):
    converted=[]
    for msg in messages:
        content=msg['content']
        if isinstance(content,str) and '<image>' in content:
            parts=content.split('<image>')
            blocks=[]
            for i,part in enumerate(parts):
                if i>0:
                    blocks.append({'type':'image'})
                if part:
                    blocks.append({'type':'text','text':part.lstrip('\n')})
            converted.append({'role':msg['role'],'content':blocks})
        else:
            converted.append(msg)
    return converted


def prompt_from_sample(processor, sample):
    messages=to_qwen_messages(sample['messages'][:-1])
    return apply_chat_template_no_think(processor, messages, tokenize=False, add_generation_prompt=True)


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


def parse_boxes(text):
    boxes=[]
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', text):
        boxes.append(tuple(map(int,m.groups())))
    return boxes


def gt_bbox_from_mask(mask_path):
    if not mask_path:
        return None,(968,968)
    im=Image.open(mask_path).convert('L')
    return im.getbbox(), im.size


def scale_box(b,size):
    w,h=size
    x1,y1,x2,y2=b
    if max(abs(x1),abs(y1),abs(x2),abs(y2)) <= 1000:
        x1=round(x1/1000*w); x2=round(x2/1000*w)
        y1=round(y1/1000*h); y2=round(y2/1000*h)
    x1=max(0,min(w,x1)); x2=max(0,min(w,x2))
    y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
    return (x1,y1,x2,y2)


def iou(a,b):
    if a is None or b is None:
        return 0.0
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,ax2-ax1)*max(0,ay2-ay1)
    bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/(aa+bb-inter+1e-9)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_path', default='/home/chang/models/Qwen3.5-2B')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_new_tokens', type=int, default=96)
    args=ap.parse_args()

    rows=read_jsonl(args.data, args.limit)
    processor=AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    inject_sve(model, proj_init_std=1e-4)
    # Transformers generation validates kwargs before forward; SVE consumes
    # image_seqlens_per_sample through a forward pre-hook, so allow it here.
    model._validate_model_kwargs = lambda model_kwargs: None
    state=torch.load(Path(args.ckpt)/'trainable_state.pt', map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('loaded ckpt', args.ckpt, 'state_tensors', len(state), 'missing', len(missing), 'unexpected', len(unexpected), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hits=0; defect_n=0; no_defect_n=0; false_alarm=0; ious=[]; parse_ok=0
    with open(args.out,'w',encoding='utf-8') as f:
        for idx,sample in enumerate(rows,1):
            prompt=prompt_from_sample(processor,sample)
            images=load_images(sample['images'])
            inputs=processor(text=[prompt], images=images, return_tensors='pt')
            inputs={k:(v.to(args.device) if hasattr(v,'to') else v) for k,v in inputs.items()}
            with torch.no_grad():
                out_ids=model.generate(**inputs, image_seqlens_per_sample=[2], max_new_tokens=args.max_new_tokens, do_sample=False)
            new_ids=out_ids[:, inputs['input_ids'].shape[1]:]
            response=processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            boxes=parse_boxes(response)
            if boxes: parse_ok += 1
            meta=sample.get('meta',{})
            has_defect=bool(meta.get('has_defect'))
            best_iou=0.0
            pred_boxes=[]
            image_size=images[1].size
            pred_boxes=[scale_box(b,image_size) for b in boxes]
            if has_defect:
                defect_n += 1
                gt,size=gt_bbox_from_mask(meta.get('mask'))
                best_iou=max([iou(b,gt) for b in pred_boxes], default=0.0)
                if best_iou >= 0.1:
                    hits += 1
                ious.append(best_iou)
            else:
                no_defect_n += 1
                if boxes or response.strip().lower() != 'no obvious defect':
                    false_alarm += 1
            rec={'idx':idx,'sample_id':meta.get('sample_id'),'product':meta.get('product'),'has_defect':has_defect,'response':response,'pred_boxes':pred_boxes,'best_iou':best_iou}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(json.dumps(rec,ensure_ascii=False), flush=True)

    summary={
        'data': args.data,
        'ckpt': args.ckpt,
        'n': len(rows),
        'defect_n': defect_n,
        'no_defect_n': no_defect_n,
        'parse_ok': parse_ok,
        'hit_at_0_1': hits,
        'hit_rate_at_0_1': hits/max(1,defect_n),
        'mean_iou': sum(ious)/max(1,len(ious)),
        'false_alarm': false_alarm,
        'false_alarm_rate': false_alarm/max(1,no_defect_n),
    }
    Path(args.out).with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY', json.dumps(summary,ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
