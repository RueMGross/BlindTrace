"""Analyse BlindTrace tracings: unblind, size, consensus, agreement.

Reads the unblinding key(s) made by prepare.py and the tracings saved by the server
(STATE/ann/<annotator>/<token>.json), maps every traced outline back onto its original
image and writes to OUT/:
  particles_all.csv        one row per traced outline, original image coordinates
  images.csv               one row per annotator x task (status, exclusion reason, counts)
  annotators.csv           per annotator x group: images, particles, median size
  intra_rater.csv          per annotator: each image traced twice (hidden repeat, or anchor
                           from an earlier round) vs its original (+ intra_rater_pairs.csv)
  inter_rater.csv          pairwise F1, ICC(2,1) of counts and sizes, Fleiss' kappa of
                           approve/exclude (>= 2 annotators)
  consensus_particles.csv  particles traced by a majority of annotators (>= 2 annotators)
  particle_agreement.csv   every outline with the number of annotators who traced it
  groups.csv               per group: D10/D50/D90 with 95 % CIs (images resampled)
  fig_groups.png           size distribution per group
  SUMMARY.md               key numbers and a methods paragraph

Method. Two outlines on the same image are the same particle if their intersection over
union (IoU, filled polygons rasterised at the image's pixels) is >= --iou (default 0.3):
outlines of two annotators are matched one-to-one by the Hungarian algorithm maximising
total IoU, and matched outlines of all annotators are joined into particles (connected
components). An image enters the consensus if a majority of annotators approved it; a
consensus particle is one traced by a majority of all annotators, and its diameter is
the mean of their diameters. Diameter = equivalent circle diameter from the polygon area
(Green's theorem); outlines that reach the image border are excluded from sizes but
count as particles. Confidence intervals resample images, not particles.

Usage: python analyze.py --key private/key.json --ann state/ann [--build build] [--out out]
       (one --key and --ann per round; --build checks the coordinate mapping pixel by pixel)
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon as rasterize

import prepare

SEED = 1          # bootstrap: same data give the same CI in every table


# ------------------------------------------------------------------ geometry
def served_to_orig(pts, k90, flip, h, w):
    """Served-image points -> original-image points (continuous coordinates, pixel i
    spanning [i, i+1)). (h, w) is the original shape. One np.rot90 step on an array of
    width ww maps (x, y) to (y, ww - x); the mirror maps x to ws - x."""
    x, y = np.asarray(pts, float).T.copy()
    shapes, hh, ww = [], h, w
    for _ in range(k90 % 4):
        shapes.append((hh, ww))
        hh, ww = ww, hh
    if flip:
        x = ww - x
    for hh, ww in reversed(shapes):
        x, y = ww - y, x
    return np.column_stack([x, y])


def poly_moments(cx, cy):
    """Area and major/minor axis of a closed polygon from the second area moments of the
    filled region (Green's theorem), independent of vertex spacing. For an ellipse with
    semi-axes a, b: major = 2a, minor = 2b."""
    x, y = np.asarray(cx, float), np.asarray(cy, float)
    x, y = x - x.mean(), y - y.mean()
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a2 = cr.sum()
    if abs(a2) < 1e-12:
        return 0.0, 0.0, 0.0
    area = a2 / 2
    gx = np.sum((x + xn) * cr) / (6 * area)
    gy = np.sum((y + yn) * cr) / (6 * area)
    sxx = np.sum((x * x + x * xn + xn * xn) * cr) / 12 / area - gx * gx
    syy = np.sum((y * y + y * yn + yn * yn) * cr) / 12 / area - gy * gy
    sxy = np.sum((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cr) / 24 / area - gx * gy
    ev = np.sort(np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]])))[::-1]
    return abs(area), 4 * np.sqrt(max(ev[0], 0.0)), 4 * np.sqrt(max(ev[1], 1e-12))


def shape_nm(p, px_nm):
    a, major, minor = poly_moments(p[:, 0], p[:, 1])
    return {'d_nm': 2 * np.sqrt(a / np.pi) * px_nm, 'major_nm': major * px_nm, 'minor_nm': minor * px_nm,
            'aspect': major / max(minor, 1e-9)}


def centroid(p):
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(0)
    return np.array([np.sum((x + xn) * cr), np.sum((y + yn) * cr)]) / (6 * a)


def iou(p, q):
    if (p[:, 0].max() < q[:, 0].min() or q[:, 0].max() < p[:, 0].min() or
            p[:, 1].max() < q[:, 1].min() or q[:, 1].max() < p[:, 1].min()):
        return 0.0
    lo = np.floor(np.minimum(p.min(0), q.min(0))) - 1
    hi = np.ceil(np.maximum(p.max(0), q.max(0))) + 2
    shp = (int(hi[1] - lo[1]), int(hi[0] - lo[0]))
    a, b = np.zeros(shp, bool), np.zeros(shp, bool)
    # pixel i spans [i, i+1): its centre i + 0.5 is the rasteriser's integer i
    a[rasterize(p[:, 1] - lo[1] - 0.5, p[:, 0] - lo[0] - 0.5, shp)] = True
    b[rasterize(q[:, 1] - lo[1] - 0.5, q[:, 0] - lo[0] - 0.5, shp)] = True
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def match_iou(A, B, thr):
    """One-to-one matching maximising total IoU (Hungarian); pairs with IoU >= thr."""
    if not len(A) or not len(B):
        return []
    M = np.array([[iou(p, q) for q in B] for p in A])
    r, c = linear_sum_assignment(-M)
    return [(i, j, M[i, j]) for i, j in zip(r, c) if M[i, j] >= thr]


# ------------------------------------------------------------------ statistics
def boot_quantile_ci(groups, qs, n=2000):
    """95 % CIs of pooled quantiles (percent), resampling images with replacement."""
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return [(np.nan, np.nan)] * len(qs)
    rng = np.random.default_rng(SEED)
    B = np.array([np.percentile(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), qs)
                  for _ in range(n)])
    return [tuple(np.percentile(B[:, k], [2.5, 97.5])) for k in range(len(qs))]


def bland_altman(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) < 2:
        return {'n_pairs': len(d), 'mean_diff_nm': d.mean() if len(d) else np.nan, 'sd_diff_nm': np.nan,
                'loa_lo_nm': np.nan, 'loa_hi_nm': np.nan}
    m, s = d.mean(), d.std(ddof=1)
    return {'n_pairs': len(d), 'mean_diff_nm': m, 'sd_diff_nm': s, 'loa_lo_nm': m - 1.96 * s, 'loa_hi_nm': m + 1.96 * s}


def cv_within(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return np.nan
    return float(np.sqrt(np.mean(((a - b) ** 2 / 2) / ((a + b) / 2) ** 2)))


def icc_2_1(Y):
    """ICC(2,1), two-way random effects, absolute agreement, single rater
    (Shrout & Fleiss 1979). Y: subjects x raters, no missing values."""
    Y = np.asarray(Y, float)
    if Y.ndim != 2 or Y.shape[0] < 2 or Y.shape[1] < 2:
        return np.nan
    n, k = Y.shape
    g, rm, cm = Y.mean(), Y.mean(1), Y.mean(0)
    msr = k * np.sum((rm - g) ** 2) / (n - 1)
    msc = n * np.sum((cm - g) ** 2) / (k - 1)
    mse = np.sum((Y - rm[:, None] - cm[None, :] + g) ** 2) / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def fleiss_kappa(table):
    """Fleiss' kappa for a subjects x categories count table (equal raters per subject)."""
    t = np.asarray(table, float)
    if not len(t):
        return np.nan
    n = t.sum(1)[0]
    p_j = t.sum(0) / t.sum()
    P_i = (np.sum(t * t, 1) - n) / (n * (n - 1))
    pe = np.sum(p_j ** 2)
    return (P_i.mean() - pe) / (1 - pe) if pe < 1 else np.nan


def prf(tp, n_b, n_a):
    """Precision (of b against a), recall, F1."""
    p = tp / n_b if n_b else np.nan
    r = tp / n_a if n_a else np.nan
    return p, r, (2 * tp / (n_a + n_b) if n_a + n_b else np.nan)


# ------------------------------------------------------------------ main
def check_mapping(key, builds):
    """Rebuild every served image from the original and the key, and check pixel by pixel
    that served == original at the positions served_to_orig maps to."""
    n_px = 0
    cache = {}
    for t, k in key.items():
        if k['image_id'] not in cache:
            cache[k['image_id']] = prepare.prepared(k)
        o = cache[k['image_id']]
        assert o.shape == (k['orig_h'], k['orig_w']), f'{t}: original size changed'
        src = [os.path.join(b, 'images', f'{t}.png') for b in builds if os.path.exists(os.path.join(b, 'images', f'{t}.png'))]
        assert src, f'{t}: served image not found in {builds}'
        s = np.asarray(prepare.Image.open(src[0]).convert('L'))
        assert np.array_equal(s, prepare.transform(o, k['k90'], k['flip'])), f'{t}: served image not reproduced'
        r, c = np.indices(s.shape)
        q = served_to_orig(np.column_stack([c.ravel() + 0.5, r.ravel() + 0.5]), k['k90'], k['flip'], *o.shape)
        xo, yo = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
        assert xo.min() >= 0 and yo.min() >= 0 and xo.max() < o.shape[1] and yo.max() < o.shape[0]
        assert np.array_equal(o[yo, xo], s.ravel()), f'{t}: mapped pixel values differ'
        n_px += s.size
    return n_px


def main(argv=None):
    ap = argparse.ArgumentParser(description='Analyse BlindTrace tracings.')
    ap.add_argument('--key', action='append', required=True, help='unblinding key of a round (repeat per round)')
    ap.add_argument('--ann', action='append', required=True, help='tracings folder STATE/ann of a round')
    ap.add_argument('--build', action='append', default=[], help='task set folder(s): check the mapping')
    ap.add_argument('--out', default='out')
    ap.add_argument('--annotators', default='', help='comma-separated initials (default: all who finished)')
    ap.add_argument('--include-partial', action='store_true', help='also use annotators with unfinished tasks')
    ap.add_argument('--iou', type=float, default=0.3, help='IoU for two outlines to be the same particle')
    a = ap.parse_args(argv)
    OUT, THR = a.out, a.iou
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    key = {}
    for kf in a.key:
        k = json.load(open(kf))
        assert not set(k) & set(key), f'{kf}: token reused across rounds'
        key.update(k)
    for v in key.values():
        if v.get('anchor_of'):         # anchor: earlier-round image shown again = cross-round repeat
            assert v['anchor_of'] in key, f'anchor_of {v["anchor_of"]}: pass the key of that round too'
            v['repeat_of'] = v['anchor_of']
    orig_tokens = [t for t, k in key.items() if k['repeat_of'] is None]
    rep_tokens = [t for t, k in key.items() if k['repeat_of'] is not None]
    n_anchor = sum(bool(k.get('anchor_of')) for k in key.values())
    groups = sorted({k['group'] for k in key.values()})
    print(f'{len(key)} tasks: {len(orig_tokens)} images + {len(rep_tokens) - n_anchor} repeats + {n_anchor} anchors; '
          f'groups: {", ".join(g or "(none)" for g in groups)}')
    n_px = check_mapping(key, a.build) if a.build else 0
    if n_px:
        print(f'mapping check: {len(key)} tasks, {n_px} pixels, served == original at mapped positions')

    # ---- annotators
    found = sorted({d for ad in a.ann if os.path.isdir(ad) for d in os.listdir(ad) if os.path.isdir(os.path.join(ad, d))})
    want = [x.strip().upper() for x in a.annotators.split(',') if x.strip()]
    RAW, annotators = {}, []
    for name in found:
        recs = {os.path.basename(f)[:-5]: json.load(open(f)) for ad in a.ann for f in glob.glob(os.path.join(ad, name, '*.json'))}
        recs = {t: r for t, r in recs.items() if t in key}
        fin = sum(r['status'] != 'draft' for r in recs.values())
        if want and name not in want:
            continue
        if fin == 0 or (fin < len(key) and not a.include_partial):
            print(f'skip {name}: {fin}/{len(key)} tasks finished')
            continue
        RAW[name] = recs
        annotators.append(name)
    if not annotators:
        sys.exit('no annotator to analyse')
    print('annotators:', ', '.join(f'{n} ({sum(r["status"] != "draft" for r in RAW[n].values())}/{len(key)})' for n in annotators))

    # ---- particles and tasks
    prow, trow, POLY, dmax = [], [], {}, 0.0
    for name in annotators:
        for t, k in key.items():
            r = RAW[name].get(t)
            st = r['status'] if r else 'missing'
            cs = (r.get('contours') or []) if r else []
            dur = (pd.Timestamp(r['t_saved']) - pd.Timestamp(r['t_open'])).total_seconds() \
                if r and r.get('t_open') and r.get('t_saved') else np.nan
            trow.append({'annotator': name, 'token': t, 'image_id': k['image_id'], 'group': k['group'], 'round': k['round'],
                         'repeat': k['repeat_of'] is not None, 'repeat_of': k['repeat_of'], 'status': st,
                         'reason': r.get('reason') if r else None, 'n_particles': len(cs),
                         'n_edge': sum(bool(c.get('edge')) for c in cs), 'n_unsure': sum(c.get('label') == 'unsure' for c in cs),
                         'area_um2': k['orig_w'] * k['orig_h'] * (k['nm_per_px'] / 1000) ** 2, 'seconds_open': dur})
            for i, c in enumerate(cs):
                p = served_to_orig(c['pts'], k['k90'], k['flip'], k['orig_h'], k['orig_w'])
                s = shape_nm(p, k['nm_per_px'])
                if c.get('d_nm') is not None:
                    dmax = max(dmax, abs(s['d_nm'] - c['d_nm']))
                cx, cy = centroid(p)
                POLY[(name, t, i)] = p
                prow.append({'annotator': name, 'token': t, 'idx': i, 'image_id': k['image_id'], 'group': k['group'],
                             'repeat': k['repeat_of'] is not None, 'task_status': st, 'label': c.get('label'),
                             'edge': bool(c.get('edge')), **s, 'd_nm_app': c.get('d_nm'), 'cx': cx, 'cy': cy, 'n_pts': len(p)})
    parts = pd.DataFrame(prow, columns=['annotator', 'token', 'idx', 'image_id', 'group', 'repeat', 'task_status', 'label',
                                        'edge', 'd_nm', 'major_nm', 'minor_nm', 'aspect', 'd_nm_app', 'cx', 'cy', 'n_pts'])
    imgs = pd.DataFrame(trow)
    print(f'{len(parts)} outlines; app vs recomputed diameter: max deviation {dmax:.3f} nm')
    assert dmax < 0.5, 'diameters stored by the app differ from the recomputed ones: wrong key for these tracings?'
    parts.to_csv(os.path.join(OUT, 'particles_all.csv'), index=False)
    imgs.to_csv(os.path.join(OUT, 'images.csv'), index=False)

    def polys(name, t):
        g = parts[(parts.annotator == name) & (parts.token == t)]
        return g, [POLY[(name, t, i)] for i in g.idx]

    done = imgs[imgs.status == 'done']
    orig_done = done[~done.repeat]
    orig_parts = parts[~parts.repeat & (parts.task_status == 'done')]

    # ---- 1. per annotator x group
    rows = []
    for name in annotators:
        for grp in groups:
            ti = imgs[(imgs.annotator == name) & (imgs.group == grp) & ~imgs.repeat]
            di = ti[ti.status == 'done']
            g = orig_parts[(orig_parts.annotator == name) & (orig_parts.group == grp)]
            sz = g[~g.edge]
            (lo, hi), = boot_quantile_ci([x.d_nm.values for _, x in sz.groupby('image_id')], [50])
            rows.append({'annotator': name, 'group': grp, 'n_images': len(ti), 'n_approved': len(di),
                         'n_excluded': int((ti.status == 'excluded').sum()),
                         'exclusion_reasons': '; '.join(f'{k}: {v}' for k, v in ti[ti.status == 'excluded'].reason.value_counts().items()),
                         'n_particles': len(g), 'n_edge': int(g.edge.sum()), 'n_unsure': int((g.label == 'unsure').sum()),
                         'particles_per_image': len(g) / max(len(di), 1),
                         'particles_per_um2': len(g) / di.area_um2.sum() if len(di) else np.nan,
                         'n_sized': len(sz), 'median_nm': sz.d_nm.median(), 'median_ci_lo': lo, 'median_ci_hi': hi,
                         'q25_nm': sz.d_nm.quantile(0.25), 'q75_nm': sz.d_nm.quantile(0.75), 'mean_nm': sz.d_nm.mean()})
    asum = pd.DataFrame(rows)
    asum.to_csv(os.path.join(OUT, 'annotators.csv'), index=False)

    # ---- 2. intra-rater: the same image traced twice by the same annotator
    rows, pair_rows = [], []
    for name in annotators:
        for t in rep_tokens:
            to = key[t]['repeat_of']
            so = imgs[(imgs.annotator == name) & (imgs.token == to)].status.iloc[0]
            sr = imgs[(imgs.annotator == name) & (imgs.token == t)].status.iloc[0]
            kind = 'cross-round' if key[t].get('anchor_of') else 'repeat'
            row = {'annotator': name, 'kind': kind, 'image_id': key[t]['image_id'], 'group': key[t]['group'],
                   'token_orig': to, 'token_rep': t, 'status_orig': so, 'status_rep': sr, 'status_agree': so == sr}
            if so == sr == 'done':
                go, po = polys(name, to)
                gr, pr = polys(name, t)
                m = match_iou(po, pr, THR)
                _, _, f1 = prf(len(m), len(gr), len(go))
                row.update({'n_orig': len(go), 'n_rep': len(gr), 'n_matched': len(m), 'f1': f1,
                            'count_diff_rep_minus_orig': len(gr) - len(go)})
                for i, j, v in m:
                    a_, b_ = go.iloc[i], gr.iloc[j]
                    pair_rows.append({'annotator': name, 'kind': kind, 'image_id': a_.image_id, 'iou': v,
                                      'a': a_.d_nm, 'b': b_.d_nm, 'edge': bool(a_.edge or b_.edge)})
            rows.append(row)
    intra = pd.DataFrame(rows)
    ipairs = pd.DataFrame(pair_rows, columns=['annotator', 'kind', 'image_id', 'iou', 'a', 'b', 'edge'])
    pooled = []
    for name in annotators:
        for kind in ('repeat', 'cross-round'):
            ik = intra[(intra.annotator == name) & (intra.kind == kind)] if len(intra) else intra
            if not len(ik):
                continue
            g = ik[(ik.status_orig == 'done') & (ik.status_rep == 'done')]
            pk = ipairs[(ipairs.annotator == name) & (ipairs.kind == kind)]
            pp = pk[~pk.edge]
            _, _, f1 = prf(g.n_matched.sum(), g.n_rep.sum(), g.n_orig.sum()) if len(g) else (np.nan,) * 3
            pooled.append({'annotator': name, 'kind': kind, 'image_id': 'POOLED', 'n_image_pairs': len(ik),
                           'status_agree': int(ik.status_agree.sum()), 'n_orig': g.n_orig.sum() if len(g) else 0,
                           'n_rep': g.n_rep.sum() if len(g) else 0, 'n_matched': g.n_matched.sum() if len(g) else 0,
                           'f1': f1, 'median_iou': pk.iou.median(), **bland_altman(pp.a, pp.b),
                           'cv_within': cv_within(pp.a, pp.b)})
    intra = pd.concat([intra, pd.DataFrame(pooled)], ignore_index=True)
    intra.to_csv(os.path.join(OUT, 'intra_rater.csv'), index=False)
    ipairs.to_csv(os.path.join(OUT, 'intra_rater_pairs.csv'), index=False)

    # ---- 3. inter-rater and consensus: an image enters if a majority approved it; a
    # consensus particle is traced by a majority of all annotators. Pairwise F1 on the images
    # both approved; ICC(2,1) on images (and, for size, particles) all approved / traced.
    inter = None
    if len(annotators) >= 2:
        need = len(annotators) // 2 + 1                      # 2 of 2, 2 of 3, 3 of 4
        appr = {n: set(orig_done[orig_done.annotator == n].token) for n in annotators}
        shared = set.intersection(*appr.values())
        major = sorted(t for t in orig_tokens if sum(t in appr[n] for n in annotators) >= need)
        rows, cons, icc_size, icc_cnt, agree = [], [], [], [], []
        for t in major:
            k = key[t]
            who_ok = [n for n in annotators if t in appr[n]]
            G = {n: polys(n, t) for n in who_ok}
            if t in shared:
                icc_cnt.append([len(G[n][0]) for n in annotators])
            nodes = [(n, i) for n in who_ok for i in range(len(G[n][0]))]
            parent = {v: v for v in nodes}

            def find(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v
            for x, na in enumerate(who_ok):
                for nb in who_ok[x + 1:]:
                    m = match_iou(G[na][1], G[nb][1], THR)
                    rows.append({'pair': f'{na}-{nb}', 'token': t, 'image_id': k['image_id'], 'group': k['group'],
                                 'n_a': len(G[na][0]), 'n_b': len(G[nb][0]), 'n_matched': len(m)})
                    for i, j, _ in m:
                        parent[find((na, i))] = find((nb, j))
            comps = {}
            for v in nodes:
                comps.setdefault(find(v), []).append(v)
            for members in comps.values():
                who = sorted({n for n, _ in members})
                rs = [G[n][0].iloc[i] for n, i in members]
                dper = {n: np.mean([G[n][0].iloc[i].d_nm for m_, i in members if m_ == n]) for n in who}
                edge = any(r_.edge for r_ in rs)
                for (n, i), r_ in zip(members, rs):
                    agree.append({'annotator': n, 'token': t, 'group': k['group'], 'd_nm': r_.d_nm, 'edge': bool(r_.edge),
                                  'n_found': len(who), 'n_approved': len(who_ok)})
                if len(who) == len(annotators) and not edge:
                    icc_size.append([dper[n] for n in annotators])
                if len(who) >= need:
                    cons.append({'token': t, 'image_id': k['image_id'], 'group': k['group'], 'round': k['round'],
                                 'n_found': len(who), 'n_approved': len(who_ok), 'found_by': ','.join(who), 'edge': edge,
                                 'd_nm': float(np.mean(list(dper.values()))), 'd_min_nm': min(dper.values()),
                                 'd_max_nm': max(dper.values()),
                                 'cx': np.mean([r_.cx for r_ in rs]), 'cy': np.mean([r_.cy for r_ in rs])})
        ir = pd.DataFrame(rows, columns=['pair', 'token', 'image_id', 'group', 'n_a', 'n_b', 'n_matched'])
        summ = []
        for pair, g in ir.groupby('pair'):
            _, _, f1 = prf(g.n_matched.sum(), g.n_b.sum(), g.n_a.sum())
            summ.append({'pair': pair, 'n_images': len(g), 'n_a': g.n_a.sum(), 'n_b': g.n_b.sum(),
                         'n_matched': g.n_matched.sum(), 'f1': f1})
        rated = imgs[~imgs.repeat & imgs.status.isin(['done', 'excluded'])].pivot(
            index='token', columns='annotator', values='status').dropna()
        table = np.column_stack([(rated == 'done').sum(axis=1), (rated == 'excluded').sum(axis=1)])
        summ.append({'pair': 'ALL', 'n_images': len(shared), 'n_images_majority': len(major),
                     'icc21_counts_per_image': icc_2_1(icc_cnt), 'icc21_size_matched': icc_2_1(icc_size),
                     'n_particles_found_by_all': len(icc_size), 'fleiss_kappa_exclusion': fleiss_kappa(table),
                     'n_images_rated': len(rated), 'n_consensus': len(cons)})
        inter = pd.DataFrame(summ)
        inter.to_csv(os.path.join(OUT, 'inter_rater.csv'), index=False)
        ir.to_csv(os.path.join(OUT, 'inter_rater_images.csv'), index=False)
        consensus = pd.DataFrame(cons, columns=['token', 'image_id', 'group', 'round', 'n_found', 'n_approved', 'found_by',
                                                'edge', 'd_nm', 'd_min_nm', 'd_max_nm', 'cx', 'cy'])
        consensus.to_csv(os.path.join(OUT, 'consensus_particles.csv'), index=False)
        pd.DataFrame(agree).to_csv(os.path.join(OUT, 'particle_agreement.csv'), index=False)
        ref = consensus[~consensus.edge]
        ref_src = f'consensus of {", ".join(annotators)} (traced by >= {need}, images approved by >= {need})'
        ref_imgs = major
    else:
        name = annotators[0]
        ref = orig_parts[(orig_parts.annotator == name) & ~orig_parts.edge]
        ref_src = f'annotator {name} (single annotator)'
        ref_imgs = sorted(orig_done[orig_done.annotator == name].token)

    # ---- 4. size per group: D10 / D50 / D90 with 95 % CIs resampling images
    rows = []
    for grp in groups:
        g = ref[ref.group == grp]
        r = {'group': grp, 'n_images_approved': sum(key[t]['group'] == grp for t in ref_imgs),
             'n_images_with_particles': g.image_id.nunique(), 'n_particles': len(g)}
        if len(g):
            q = np.percentile(g.d_nm, [10, 50, 90])
            ci = boot_quantile_ci([x.d_nm.values for _, x in g.groupby('image_id')], [10, 50, 90])
            for pc, v, (lo, hi) in zip((10, 50, 90), q, ci):
                r.update({f'D{pc}_nm': v, f'D{pc}_ci_lo': lo, f'D{pc}_ci_hi': hi})
            r.update({'mean_nm': g.d_nm.mean(), 'sd_nm': g.d_nm.std(), 'geo_mean_nm': float(np.exp(np.log(g.d_nm).mean()))})
        for n in annotators:
            x = asum[(asum.annotator == n) & (asum.group == grp)]
            r[f'D50_{n}_nm'] = float(x.median_nm.iloc[0]) if len(x) else np.nan
        rows.append(r)
    gsum = pd.DataFrame(rows)
    gsum.to_csv(os.path.join(OUT, 'groups.csv'), index=False)
    fig_groups(ref, gsum, groups, ref_src, OUT)
    write_summary(OUT, annotators, key, n_px, asum, intra, inter, gsum, ref_src, THR, len(parts))
    print('done ->', OUT)


def fig_groups(ref, gsum, groups, ref_src, out):
    """Size distribution per group: KDE of log diameter, box (median, IQR, D10-D90)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        return
    gs = [g for g in groups if (ref.group == g).sum() >= 3]
    if not gs:
        return
    lo, hi = max(1.0, ref.d_nm.min() / 1.5), ref.d_nm.max() * 1.5
    x = np.linspace(np.log10(lo), np.log10(hi), 500)
    fig, ax = plt.subplots(figsize=(6, 0.9 + 0.9 * len(gs)))
    for i, g in enumerate(gs):
        d = ref[ref.group == g].d_nm.values
        base = len(gs) - 1 - i
        y = gaussian_kde(np.log10(d))(x)
        y = 0.62 * y / y.max()
        ax.fill_between(10 ** x, base + 0.22, base + 0.22 + y, color='#2a78d6', alpha=0.28, lw=0)
        ax.plot(10 ** x, base + 0.22 + y, color='#2a78d6', lw=1.3)
        q = np.percentile(d, [10, 25, 50, 75, 90])
        ax.plot([q[0], q[4]], [base + 0.1] * 2, color='k', lw=0.9)
        ax.plot([q[1], q[3]], [base + 0.1] * 2, color='k', lw=4)
        ax.plot(q[2], base + 0.1, 'o', ms=4.5, mfc='white', mec='k')
        ax.text(10 ** x[-1], base + 0.6, f'{q[2]:.0f} nm\nn = {len(d)}', ha='right', va='center', fontsize=8)
        ax.text(lo, base + 0.45, f'{g or "(no group)"}  ', ha='right', va='center', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, len(gs))
    ax.set_yticks([])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('Diameter (nm)')
    ax.set_title(f'Reference: {ref_src}; outlines at the image edge excluded', fontsize=7, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_groups.png'), dpi=200)
    plt.close(fig)


def write_summary(out, annotators, key, n_px, asum, intra, inter, gsum, ref_src, thr, n_outlines):
    f = lambda v, d=0: '' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'  # noqa: E731
    L = ['# BlindTrace analysis', '',
         f'Annotators: {", ".join(annotators)}. Tasks: {len(key)}; outlines: {n_outlines}.',
         f'Mapping check: {f"{n_px} pixels, served == original at every mapped position" if n_px else "not run (no --build)"}.',
         f'Size reference: {ref_src}.', '',
         '## Size per group (nm; 95 % CI resampling images)', '',
         '| Group | Images | Particles | D10 | D50 (95 % CI) | D90 | ' + ' | '.join(f'D50 {n}' for n in annotators) + ' |',
         '|---|---|---|---|---|---|' + '---|' * len(annotators)]
    for r in gsum.itertuples():
        g = r._asdict()
        L.append(f'| {r.group or "(none)"} | {r.n_images_approved} | {r.n_particles} | {f(g.get("D10_nm"))} | '
                 f'{f(g.get("D50_nm"))} ({f(g.get("D50_ci_lo"))}-{f(g.get("D50_ci_hi"))}) | {f(g.get("D90_nm"))} | '
                 + ' | '.join(f(g.get(f'D50_{n}_nm')) for n in annotators) + ' |')
    if inter is not None:
        a = inter[inter.pair == 'ALL'].iloc[0]
        L += ['', '## Agreement between annotators', '',
              f'- Images approved by all: {a.n_images}; by a majority: {a.n_images_majority}; consensus particles: {a.n_consensus}',
              f'- Size of the same particle, ICC(2,1): {f(a.icc21_size_matched, 2)} ({a.n_particles_found_by_all} particles traced by all)',
              f'- Particles per image, ICC(2,1): {f(a.icc21_counts_per_image, 2)}',
              f'- Approve vs exclude, Fleiss kappa: {f(a.fleiss_kappa_exclusion, 2)}']
        for r in inter[inter.pair != 'ALL'].itertuples():
            L.append(f'- {r.pair}: F1 {f(r.f1, 2)} ({r.n_matched} matched of {r.n_a} / {r.n_b}, {r.n_images} images)')
    pooled = intra[intra.image_id == 'POOLED'] if len(intra) else intra
    if len(pooled):
        L += ['', '## Same annotator, same image twice', '',
              '| Annotator | Kind | Image pairs | F1 | Size difference, mean (nm) | Within-particle CV |', '|---|---|---|---|---|---|']
        for r in pooled.itertuples():
            L.append(f'| {r.annotator} | {r.kind} | {r.n_image_pairs} | {f(r.f1, 2)} | {f(r.mean_diff_nm, 1)} | '
                     f'{f(100 * r.cv_within, 1)} % |')
    L += ['', '## Methods (draft)', '',
          f'Particles were traced by hand by {len(annotators)} annotators with BlindTrace, blinded to sample identity: '
          'images were shown in random order and random orientation (rotation by multiples of 90 degrees and mirroring) '
          'under opaque names, and a subset was shown twice as hidden repeats. Annotators traced the outer boundary of every '
          'particle, could mark outlines as unsure, and approved or excluded each image with a reason. Outlines were mapped '
          'back to the original images. Two outlines were taken to be the same particle if their intersection over union was '
          f'at least {thr} (one-to-one Hungarian matching between annotators, joined across annotators as connected components). '
          'Images approved by a majority of annotators were analysed; a consensus particle was one traced by a majority of '
          'annotators, with its diameter the mean of their equivalent circle diameters (from the polygon area). Particles '
          'cut by the image border were counted but excluded from sizes. 95 % confidence intervals of size quantiles were '
          'obtained by resampling images with replacement (2,000 resamples). Agreement was quantified by pairwise F1, '
          "ICC(2,1) and Fleiss' kappa, and consistency by the repeats."]
    open(os.path.join(out, 'SUMMARY.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
