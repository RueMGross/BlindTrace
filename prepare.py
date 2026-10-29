"""Build a blinded task set for BlindTrace from a list of images.

Input: a CSV manifest, one row per image:
  image        path to the image (PNG, TIFF incl. 16-bit, JPEG, ...), relative to the CSV
  nm_per_px    pixel size in nm
  group        optional: sample / condition the image belongs to (never shown to annotators)
  image_id     optional: unique name (default: the path)
  crop_top, crop_bottom, crop_left, crop_right
               optional: pixels to cut off, e.g. a data bar or a burned-in scale bar
               (anything that could identify the image or the group)

Each image is converted to grey, cropped, optionally resampled to one pixel size for
all images (--nm-per-px; otherwise pixel size or image size may reveal the group),
contrast-stretched between two percentiles to 8 bit, and served under an opaque random
token in a random orientation (rotation by 0/90/180/270 degrees, optionally mirrored).
A fraction of the images is shown a second time as hidden repeats (in a different
orientation) to measure each annotator's consistency. For a later round, images of
earlier rounds can be mixed in again as anchors (--previous, --anchors), which measures
consistency across rounds and keeps a new image set from being recognisable as such.

Outputs
  OUT/images/<token>.png, OUT/tasks.json   the task set; goes on the server
  KEY (default private/key.json)           UNBLINDING KEY: token -> image, group,
                                           orientation, preparation. Keep it off the
                                           server and away from the annotators.

Usage: python prepare.py manifest.csv [--title "Round 1"] [--repeat-frac 0.1]
       python prepare.py round2.csv --key private/key_r2.json --out build_r2 \\
                         --previous private/key.json --anchors 8 --round 2
"""
import argparse
import json
import os
import secrets
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

CROPS = ('crop_top', 'crop_bottom', 'crop_left', 'crop_right')


def load_grey(path):
    """First frame of an image as a float array; colour images are converted to grey."""
    im = Image.open(path)
    im.seek(0)
    if im.mode not in ('L', 'I', 'I;16', 'I;16B', 'I;16L', 'F'):
        im = im.convert('L')
    a = np.asarray(im, dtype=float)
    if a.ndim != 2:
        raise ValueError(f'{path}: not a single-channel image after conversion')
    return a


def prepared(entry):
    """The analysable image (8 bit, before orientation) and its pixel size, rebuilt
    from a key entry (or a manifest row turned into one by make_entry)."""
    a = load_grey(entry['image'])
    t, b, left, r = entry['crop']
    a = a[t:a.shape[0] - b, left:a.shape[1] - r]
    if entry['resample'] and entry['nm_per_px'] != entry['nm_per_px_source']:
        f = entry['nm_per_px_source'] / entry['nm_per_px']
        size = (max(1, round(a.shape[1] * f)), max(1, round(a.shape[0] * f)))
        a = np.asarray(Image.fromarray(a.astype(np.float32), 'F').resize(size, Image.Resampling.LANCZOS), dtype=float)
    lo, hi = np.percentile(a, entry['stretch'])
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).round().astype(np.uint8)


def transform(a, k90, flip):
    """Served orientation: np.rot90 by k90 quarter turns, then mirrored left-right if flip."""
    a = np.rot90(a, k90)
    return a[:, ::-1] if flip else a


def make_entry(row, base, a):
    crop = [int(row[c]) if c in row and pd.notna(row[c]) else 0 for c in CROPS]
    src = float(row['nm_per_px'])
    return {'image_id': str(row['image_id']) if 'image_id' in row and pd.notna(row['image_id']) else str(row['image']),
            'image': os.path.abspath(os.path.join(base, str(row['image']))),
            'group': str(row['group']) if 'group' in row and pd.notna(row['group']) else '',
            'crop': crop, 'nm_per_px_source': src, 'nm_per_px': a.nm_per_px or src, 'resample': bool(a.nm_per_px),
            'stretch': [a.stretch[0], a.stretch[1]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('manifest', help='CSV with columns image, nm_per_px [, group, image_id, crop_*]')
    ap.add_argument('--out', default='build', help='task set folder for the server (default: build)')
    ap.add_argument('--key', default=os.path.join('private', 'key.json'), help='unblinding key (keep private)')
    ap.add_argument('--title', default='', help='shown on the start screen, e.g. "Round 2"')
    ap.add_argument('--repeat-frac', type=float, default=0.1, help='fraction of images shown twice (default 0.1)')
    ap.add_argument('--nm-per-px', type=float, default=None, help='resample every image to this pixel size')
    ap.add_argument('--stretch', type=float, nargs=2, default=(0.5, 99.5), metavar=('LO', 'HI'),
                    help='contrast stretch between these grey-value percentiles (default 0.5 99.5)')
    ap.add_argument('--previous', nargs='*', default=[], help='keys of earlier rounds (their tokens are not reused)')
    ap.add_argument('--anchors', type=int, default=0, help='images of earlier rounds to show again')
    ap.add_argument('--anchor-groups', default='', help='comma-separated groups to draw anchors from (default all)')
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--seed', type=int, default=None, help='for orientation and repeat choice (default: random)')
    ap.add_argument('--force', action='store_true', help='overwrite an existing key (orphans its tracings)')
    a = ap.parse_args()

    if os.path.exists(a.key) and not a.force:
        sys.exit(f'{a.key} exists: this task set is already prepared (re-running would orphan its tracings; --force)')
    if os.path.isdir(a.out) and os.listdir(a.out) and not os.path.exists(os.path.join(a.out, 'tasks.json')):
        sys.exit(f'{a.out} is not empty and is not a task set: refusing to replace it')
    seed = a.seed if a.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)

    man = pd.read_csv(a.manifest)
    for c in ('image', 'nm_per_px'):
        if c not in man.columns:
            sys.exit(f'manifest needs a column "{c}"')
    base = os.path.dirname(os.path.abspath(a.manifest))
    rows = [make_entry(r, base, a) for r in man.to_dict('records')]
    ids = [e['image_id'] for e in rows]
    if len(set(ids)) != len(ids):
        sys.exit('image_id (or image path) must be unique')

    prev = {}
    for kf in a.previous:
        prev.update(json.load(open(kf)))
    anchors = []
    if a.anchors:
        groups = {g for g in a.anchor_groups.split(',') if g}
        pool = sorted((t for t, e in prev.items() if e['repeat_of'] is None and e.get('anchor_of') is None
                       and (not groups or e['group'] in groups)), key=lambda t: prev[t]['image_id'])
        if len(pool) < a.anchors:
            sys.exit(f'only {len(pool)} earlier images available as anchors')
        anchors = [str(t) for t in rng.choice(pool, a.anchors, replace=False)]

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, 'images'))
    n_rep = int(round(a.repeat_frac * len(rows)))
    rep_rows = set(rng.choice(len(rows), n_rep, replace=False).tolist())
    tasks, key = [], {}

    def emit(entry, img8, k90, flip, repeat_of, anchor_of):
        while True:
            token = secrets.token_hex(6)
            if token not in prev and token not in key:
                break
        t = transform(img8, k90, flip)
        Image.fromarray(np.ascontiguousarray(t)).save(os.path.join(a.out, 'images', f'{token}.png'), optimize=True)
        tasks.append({'token': token, 'nm_per_px': entry['nm_per_px'], 'w': int(t.shape[1]), 'h': int(t.shape[0]),
                      'after': repeat_of})
        key[token] = {**{k: entry[k] for k in ('image_id', 'image', 'group', 'crop', 'nm_per_px_source', 'nm_per_px',
                                                'resample', 'stretch')},
                      'k90': int(k90), 'flip': bool(flip), 'orig_w': int(img8.shape[1]), 'orig_h': int(img8.shape[0]),
                      'repeat_of': repeat_of, 'anchor_of': anchor_of, 'round': a.round}
        return token

    def orient(avoid=()):
        while True:
            k90, flip = int(rng.integers(4)), bool(rng.integers(2))
            if (k90, flip) not in avoid:
                return k90, flip

    for i, e in enumerate(rows):
        img8 = prepared(e)
        k90, flip = orient()
        tok = emit(e, img8, k90, flip, None, None)
        if i in rep_rows:
            emit(e, img8, *orient({(k90, flip)}), tok, None)
        print(f'{i + 1:4d}/{len(rows)} {e["image_id"]}', flush=True)
    for t in anchors:
        e = prev[t]
        seen = {(v['k90'], v['flip']) for v in prev.values() if v['image_id'] == e['image_id']}
        emit(e, prepared(e), *orient(seen), None, t)
        print(f'anchor {e["image_id"]}', flush=True)

    json.dump({'title': a.title, 'tasks': tasks}, open(os.path.join(a.out, 'tasks.json'), 'w'), indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.key)), exist_ok=True)
    json.dump(key, open(a.key, 'w'), indent=1)
    print(f'{len(tasks)} tasks: {len(rows)} images + {n_rep} repeats + {len(anchors)} anchors (seed {seed})\n'
          f'serve {a.out}/; keep {a.key} private')


if __name__ == '__main__':
    main()
