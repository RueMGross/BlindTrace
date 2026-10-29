"""Make a synthetic demo data set: EM-like images of round particles of known size.

Two groups, A (median diameter 80 nm) and B (120 nm), 6 images each, 1 nm per pixel.
Particles have a dark rim on a noisy background; some are cut by the image border. One
image is a 16-bit TIFF with a bright data bar at the bottom that prepare.py crops away
(crop_bottom), as with real microscope exports.

Outputs (in --out, default: next to this script)
  images/*.png, images/*.tif   the images
  manifest.csv                 input for prepare.py
  truth.csv                    every particle: image, group, centre, true diameter

Usage: python demo/make_demo.py [--seed 7] [--out DIR]
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE, BAR = 700, 40
GROUPS = {'A': 80.0, 'B': 120.0}


def render(rng, parts, h, w):
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    img = np.full((h, w), 0.62)
    for cx, cy, d in parts:
        r = np.hypot(xx - cx, yy - cy) - d / 2
        img -= 0.10 * expit(-r / 0.8)                             # slightly darker interior
        img -= 0.22 * np.exp(-0.5 * ((r + 2.0) / 1.6) ** 2)      # dark membrane rim
    noise = rng.normal(0, 0.06, (h, w))
    k = np.ones(3) / 3                                            # mild correlation, like real grain
    noise = np.apply_along_axis(np.convolve, 0, np.apply_along_axis(np.convolve, 1, noise, k, 'same'), k, 'same')
    return np.clip(img + noise, 0, 1)


def place(rng, n, median, h, w):
    out = []
    for _ in range(2000):
        if len(out) == n:
            break
        d = float(np.clip(rng.lognormal(np.log(median), 0.28), 30, 400))
        cx, cy = rng.uniform(-0.2 * d, w + 0.2 * d), rng.uniform(-0.2 * d, h + 0.2 * d)
        if all(np.hypot(cx - x, cy - y) > (d + e) / 2 + 12 for x, y, e in out):
            out.append((cx, cy, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    man, truth = [], []
    for g, med in GROUPS.items():
        for i in range(6):
            iid = f'{g}{i + 1}'
            parts = place(rng, int(rng.integers(8, 15)), med, SIZE, SIZE)
            img = render(rng, parts, SIZE, SIZE)
            row = {'image': '', 'nm_per_px': 1.0, 'group': g, 'image_id': iid, 'crop_bottom': 0}
            if g == 'B' and i == 0:            # 16-bit TIFF with a data bar to crop
                full = np.vstack([img, np.ones((BAR, SIZE))])
                full[SIZE + 12:SIZE + 28, 20:300:7] = 0
                row.update(image=f'images/{iid}.tif', crop_bottom=BAR)
                Image.fromarray((full * 65535).astype(np.uint16)).save(os.path.join(a.out, row['image']))
            else:
                row['image'] = f'images/{iid}.png'
                Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(a.out, row['image']))
            man.append(row)
            truth += [{'image_id': iid, 'group': g, 'cx': cx, 'cy': cy, 'd_nm': d} for cx, cy, d in parts]
    pd.DataFrame(man).to_csv(os.path.join(a.out, 'manifest.csv'), index=False)
    pd.DataFrame(truth).to_csv(os.path.join(a.out, 'truth.csv'), index=False)
    print(f'{len(man)} images, {len(truth)} particles -> {a.out}')


if __name__ == '__main__':
    main()
