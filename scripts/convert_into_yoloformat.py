import re, json, shutil, yaml, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

#config
ROOT = Path(__file__).resolve().parents[1]            

SRC = ROOT / 'data' / 'raw'                         
DST = ROOT / 'data' / 'yolo'                          

CLASSES = ['zebra', 'giraffe', 'elephant']            # 0, 1, 2
ALIASES = {'zebra': 0, 'giraffe': 1, 'elephant': 2}   

TEST_VIDEOS = ['DJI_20230719145816_0003_V_video2', 'DJI_0601_video4']
VAL_VIDEOS  = ['DJI_20230719145427_0002_V_video5', 'DJI_0117_video4', 'DJI_0204_video2']
#the rest ->train


videos = sorted(p for p in SRC.iterdir() if p.is_dir())
names = {v.name for v in videos}
for n in TEST_VIDEOS + VAL_VIDEOS:
    if n not in names:
        print(f'no found: {n}')
print(f'total: {len(videos)}')

for sp in ['train', 'val', 'test']:
    old = list((DST / 'images' / sp).glob('*.jpg')) if (DST / 'images' / sp).exists() else []


def split_of(name):
    if name in TEST_VIDEOS: return 'test'
    if name in VAL_VIDEOS:  return 'val'
    return 'train'

def labelme_to_yolo(json_path):
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    W, H = d['imageWidth'], d['imageHeight']
    lines, unknown = [], []
    for s in d.get('shapes', []):
        lb = s['label'].strip().lower()
        if lb not in ALIASES:
            unknown.append(lb); continue
        xs = [p[0] for p in s['points']]
        ys = [p[1] for p in s['points']]
        x1, x2 = max(0, min(xs)), min(W, max(xs))
        y1, y2 = max(0, min(ys)), min(H, max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f'{ALIASES[lb]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
    return lines, unknown

def frame_name(vname, stem):
    m = re.search(r'(\d+)$', stem)
    num = f'{int(m.group(1)):06d}' if m else stem
    return f'{vname}_frame_{num}'

jobs, missing_json = [], 0
per_video = collections.Counter()
for v in videos:
    img_dir, js_dir = v / 'frames', v / 'boxid'
    if not img_dir.exists() or not js_dir.exists():
        print(f' {v.name} thiếu frames/boxid'); continue
    sp = split_of(v.name)
    (DST / 'images' / sp / v.name).mkdir(parents=True, exist_ok=True)
    (DST / 'labels' / sp / v.name).mkdir(parents=True, exist_ok=True)
    for img in img_dir.glob('*.jpg'):
        js = js_dir / f'{img.stem}.json'
        if not js.exists():
            missing_json += 1; continue
        jobs.append((v.name, sp, img, js))
        per_video[v.name] += 1

print(f'image-json: {len(jobs)} | loss_image: {missing_json}')

stats = collections.Counter()
imgs_per_split = collections.Counter()
unknown_all = collections.Counter()

def process(job):
    vname, sp, img, js = job
    name = frame_name(vname, img.stem)
    out_img = DST / 'images' / sp / vname / f'{name}.jpg'
    out_txt = DST / 'labels' / sp / vname / f'{name}.txt'
    lines, unknown = labelme_to_yolo(js)
    out_txt.write_text('\n'.join(lines) + ('\n' if lines else ''))
    if not out_img.exists():                      
        shutil.copy2(img, out_img)
    return sp, lines, unknown

with ThreadPoolExecutor(max_workers=16) as ex:
    for sp, lines, unknown in tqdm(ex.map(process, jobs), total=len(jobs)):
        imgs_per_split[sp] += 1
        for l in lines:
            stats[(sp, int(l.split()[0]))] += 1
        unknown_all.update(unknown)
        
for sp in ['train', 'val', 'test']:
    print(f'{sp}: {imgs_per_split[sp]}')
print('\n=== boxes per class ===')
for sp in ['train', 'val', 'test']:
    print(sp, {CLASSES[c]: stats[(sp, c)] for c in range(len(CLASSES))})
print('\n=== images per video ===')
for sp in ['train', 'val', 'test']:
    print(f'[{sp}]')
    for v in videos:
        if split_of(v.name) == sp and per_video[v.name]:
            print(f'  {v.name}: {per_video[v.name]}')
if unknown_all:
    print('\nlabels not in ALIASES (skipped):', dict(unknown_all))
print('\nDone:', DST)
print('data.yaml:', DATA_YAML)