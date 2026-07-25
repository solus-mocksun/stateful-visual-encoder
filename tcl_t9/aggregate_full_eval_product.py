#!/usr/bin/env python3
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path


def parse_boxes(text):
    return re.findall(r'"bbox_2d"\s*:\s*\[', text or '')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(x):
    return f'{x*100:.2f}%'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--threshold', type=float, default=0.1)
    args=ap.parse_args()
    out=Path(args.outdir)
    pred_all=out/'pred_all.jsonl'
    shards=sorted(out.glob('pred_shard*.jsonl'))
    with pred_all.open('w', encoding='utf-8') as w:
        for p in shards:
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    rec=json.loads(line)
                    rec['_shard']=p.name
                    w.write(json.dumps(rec, ensure_ascii=False)+'\n')

    stats=defaultdict(lambda: {'defect':0,'miss':0,'iou_sum':0.0,'no_defect':0,'fa':0,'parse_ok':0})
    total={'defect':0,'miss':0,'iou_sum':0.0,'no_defect':0,'fa':0,'parse_ok':0}
    for rec in read_jsonl(pred_all):
        product=rec.get('product') or 'UNKNOWN'
        buckets=[stats[product], total]
        has=bool(rec.get('has_defect'))
        boxes=parse_boxes(rec.get('response',''))
        for b in buckets:
            if boxes: b['parse_ok'] += 1
            if has:
                b['defect'] += 1
                iou=float(rec.get('best_iou') or 0.0)
                b['iou_sum'] += iou
                if iou < args.threshold:
                    b['miss'] += 1
            else:
                b['no_defect'] += 1
                if boxes or (rec.get('response','').strip().lower() != 'no obvious defect'):
                    b['fa'] += 1

    rows=[]
    for product,s in sorted(stats.items()):
        defect=s['defect']; no=s['no_defect']
        hit=defect-s['miss']
        precision_den=hit+s['fa']
        rows.append({
            'product': product,
            'total_defect': defect,
            'miss': s['miss'],
            'miss_rate': s['miss']/defect if defect else 0.0,
            'recall': hit/defect if defect else 0.0,
            'mean_iou': s['iou_sum']/defect if defect else 0.0,
            'total_no_defect': no,
            'false_alarm': s['fa'],
            'false_alarm_rate': s['fa']/no if no else 0.0,
            'precision': hit/precision_den if precision_den else 1.0,
            'parse_ok': s['parse_ok'],
        })
    rows.sort(key=lambda r: (-r['miss_rate'], -r['total_defect']))

    csv_path=out/'product_summary.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames=['product','total_defect','miss','miss_rate','recall','mean_iou','total_no_defect','false_alarm','false_alarm_rate','precision','parse_ok']
        wr=csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            rr=r.copy()
            for k in ['miss_rate','recall','false_alarm_rate','precision']:
                rr[k]=pct(rr[k])
            rr['mean_iou']=f"{rr['mean_iou']:.4f}"
            wr.writerow(rr)
    (out/'product_summary.json').write_text(json.dumps({'overall': total, 'products': rows}, ensure_ascii=False, indent=2), encoding='utf-8')

    defect=total['defect']; no=total['no_defect']; hit=defect-total['miss']; precision_den=hit+total['fa']
    overall={
        'total_defect': defect,
        'miss': total['miss'],
        'miss_rate': total['miss']/defect if defect else 0.0,
        'recall': hit/defect if defect else 0.0,
        'mean_iou': total['iou_sum']/defect if defect else 0.0,
        'total_no_defect': no,
        'false_alarm': total['fa'],
        'false_alarm_rate': total['fa']/no if no else 0.0,
        'precision': hit/precision_den if precision_den else 1.0,
        'parse_ok': total['parse_ok'],
        'n': defect+no,
    }
    md=[]
    md.append('# Full Test Product Summary\n')
    md.append('## Overall\n')
    md.append(f"- Total defect: {overall['total_defect']}\n")
    md.append(f"- Miss: {overall['miss']} ({pct(overall['miss_rate'])})\n")
    md.append(f"- Recall: {pct(overall['recall'])}\n")
    md.append(f"- Mean IoU: {overall['mean_iou']:.4f}\n")
    md.append(f"- Total no-defect: {overall['total_no_defect']}\n")
    md.append(f"- False alarm: {overall['false_alarm']} ({pct(overall['false_alarm_rate'])})\n")
    md.append(f"- Precision: {pct(overall['precision'])}\n")
    md.append('\n## Products sorted by miss rate\n')
    md.append('| Product | Total Defect | Miss | Miss Rate | Recall | mIoU | No-Defect | FA | FA Rate | Precision |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:
        md.append(f"| {r['product']} | {r['total_defect']} | {r['miss']} | {pct(r['miss_rate'])} | {pct(r['recall'])} | {r['mean_iou']:.4f} | {r['total_no_defect']} | {r['false_alarm']} | {pct(r['false_alarm_rate'])} | {pct(r['precision'])} |\n")
    (out/'product_summary.md').write_text(''.join(md), encoding='utf-8')
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print('wrote', pred_all, csv_path, out/'product_summary.md')

if __name__=='__main__': main()
