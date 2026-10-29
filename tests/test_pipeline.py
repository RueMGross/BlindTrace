"""End-to-end tests: synthetic images -> prepare -> simulated annotators -> analyze, a
second round with anchors, and the server API in-process.

The simulated annotators trace the known particles of the demo images in the served
(rotated, mirrored) images, so the test checks that every outline is mapped back to the
right place and size. Run: python -m pytest tests/
"""
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analyze  # noqa: E402
import server  # noqa: E402

PY = sys.executable
# annotator: (size factor, relative noise, chance to miss a particle < 60 nm)
ANNOTATORS = {'AA': (1.00, 0.03, 0.2), 'BB': (1.04, 0.03, 0.4), 'CC': (0.97, 0.04, 0.1)}


def run(*args):
    subprocess.run([PY, *args], check=True, cwd=ROOT, capture_output=True, text=True)


def orig_to_served(pts, k90, flip, h, w):
    """Inverse of analyze.served_to_orig (an affine map), solved from three points."""
    S = np.array([[0, 0], [1, 0], [0, 1]], float)
    O = analyze.served_to_orig(S, k90, flip, h, w)
    M = np.linalg.solve(np.column_stack([O, np.ones(3)]), S)
    return np.column_stack([pts, np.ones(len(pts))]) @ M


def simulate(key, truth, ann_dir, seed, exclude=None):
    """Write tracings as the app saves them: outlines in served-image pixels."""
    rng = np.random.default_rng(seed)
    for name, (factor, noise, miss) in ANNOTATORS.items():
        os.makedirs(os.path.join(ann_dir, name), exist_ok=True)
        for t, k in key.items():
            served_w, served_h = (k['orig_h'], k['orig_w']) if k['k90'] % 2 else (k['orig_w'], k['orig_h'])
            cs = []
            for p in truth[truth.image_id == k['image_id']].itertuples():
                if p.d_nm < 60 and rng.random() < miss:
                    continue
                r = p.d_nm / 2 / k['nm_per_px'] * factor * (1 + rng.normal(0, noise))
                a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
                c = np.array([p.cx, p.cy]) / k['nm_per_px'] + rng.normal(0, 1, 2)
                pts = orig_to_served(np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)]),
                                     k['k90'], k['flip'], k['orig_h'], k['orig_w'])
                edge = bool((pts <= 0.5).any() or (pts[:, 0] >= served_w - 0.5).any() or (pts[:, 1] >= served_h - 0.5).any())
                cs.append({'label': 'particle', 'pts': pts.round(2).tolist(), 'edge': edge,
                           'd_nm': round(analyze.shape_nm(pts, k['nm_per_px'])['d_nm'], 1)})
            st = 'excluded' if exclude and (name, k['image_id']) in exclude and k['repeat_of'] is None else 'done'
            rec = {'status': st, 'contours': cs if st == 'done' else [], 'reason': 'focus' if st == 'excluded' else None,
                   't_open': '2026-01-01T10:00:00Z', 't_saved': '2026-01-01T10:01:00Z'}
            json.dump(rec, open(os.path.join(ann_dir, name, f'{t}.json'), 'w'))


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp('demo')
    run('demo/make_demo.py', '--seed', '7', '--out', str(d))
    run('prepare.py', str(d / 'manifest.csv'), '--out', str(d / 'build'), '--key', str(d / 'key.json'),
        '--title', 'Test round', '--seed', '3', '--repeat-frac', '0.25')
    return d


def test_prepare_refuses_to_overwrite_key(demo):
    r = subprocess.run([PY, 'prepare.py', str(demo / 'manifest.csv'), '--out', str(demo / 'build'), '--key',
                        str(demo / 'key.json')], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and 'exists' in r.stderr


def test_end_to_end(demo):
    key = json.load(open(demo / 'key.json'))
    truth = pd.read_csv(demo / 'truth.csv')
    assert sum(k['repeat_of'] is not None for k in key.values()) == 3
    tif = [k for k in key.values() if k['image'].endswith('.tif')][0]
    assert (tif['orig_w'], tif['orig_h']) == (700, 700)          # data bar cropped
    simulate(key, truth, demo / 'state' / 'ann', seed=1, exclude={('CC', 'A2')})
    out = demo / 'out'
    analyze.main(['--key', str(demo / 'key.json'), '--ann', str(demo / 'state' / 'ann'),
                  '--build', str(demo / 'build'), '--out', str(out)])
    cons = pd.read_csv(out / 'consensus_particles.csv')
    # every consensus particle lands on a true particle of its image, with the right size
    errs = []
    for c in cons[~cons.edge].itertuples():
        tr = truth[truth.image_id == c.image_id]
        j = np.argmin(np.hypot(tr.cx - c.cx, tr.cy - c.cy))
        assert np.hypot(tr.cx.iloc[j] - c.cx, tr.cy.iloc[j] - c.cy) < 3
        errs.append(c.d_nm / tr.d_nm.iloc[j] - 1)
    inside = truth[(truth.cx - truth.d_nm / 2 > 3) & (truth.cy - truth.d_nm / 2 > 3) &
                   (truth.cx + truth.d_nm / 2 < 697) & (truth.cy + truth.d_nm / 2 < 697)]
    assert len(errs) >= 0.9 * len(inside) and abs(np.median(errs)) < 0.03
    g = pd.read_csv(out / 'groups.csv').set_index('group')
    assert 70 < g.loc['A', 'D50_nm'] < 95 and 105 < g.loc['B', 'D50_nm'] < 140
    assert g.loc['A', 'n_images_approved'] == 6                  # A2 excluded by 1 of 3 only
    inter = pd.read_csv(out / 'inter_rater.csv').set_index('pair')
    assert inter.loc['ALL', 'icc21_size_matched'] > 0.9
    intra = pd.read_csv(out / 'intra_rater.csv')
    assert (intra[intra.image_id == 'POOLED'].f1 > 0.8).all()
    assert (out / 'SUMMARY.md').exists() and (out / 'fig_groups.png').exists()


def test_second_round_with_anchors(demo, tmp_path):
    man = pd.read_csv(demo / 'manifest.csv').head(4)
    man['image_id'] = man.image_id + '_r2'
    man['image'] = [str(demo / p) for p in man.image]
    man.to_csv(tmp_path / 'r2.csv', index=False)
    run('prepare.py', str(tmp_path / 'r2.csv'), '--out', str(tmp_path / 'build_r2'), '--key', str(tmp_path / 'key_r2.json'),
        '--previous', str(demo / 'key.json'), '--anchors', '3', '--round', '2', '--seed', '5', '--repeat-frac', '0')
    k1, k2 = json.load(open(demo / 'key.json')), json.load(open(tmp_path / 'key_r2.json'))
    anchors = [v for v in k2.values() if v['anchor_of']]
    assert len(anchors) == 3 and not set(k1) & set(k2)
    for v in anchors:                                            # new orientation
        assert (v['k90'], v['flip']) != (k1[v['anchor_of']]['k90'], k1[v['anchor_of']]['flip'])
    truth = pd.read_csv(demo / 'truth.csv')
    truth = pd.concat([truth, truth.assign(image_id=truth.image_id + '_r2')])
    simulate(k1, truth, tmp_path / 'ann1', seed=2)
    simulate(k2, truth, tmp_path / 'ann2', seed=3)
    analyze.main(['--key', str(demo / 'key.json'), '--key', str(tmp_path / 'key_r2.json'),
                  '--ann', str(tmp_path / 'ann1'), '--ann', str(tmp_path / 'ann2'),
                  '--build', str(demo / 'build'), '--build', str(tmp_path / 'build_r2'), '--out', str(tmp_path / 'out')])
    intra = pd.read_csv(tmp_path / 'out' / 'intra_rater.csv')
    cross = intra[(intra.image_id == 'POOLED') & (intra.kind == 'cross-round')]
    assert len(cross) == 3 and (cross.n_image_pairs == 3).all() and (cross.f1 > 0.8).all()


def test_server_api(demo, tmp_path):
    server.setup(str(demo / 'build'), str(tmp_path / 'state'), prefix='/bt')
    httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}/bt'
    get = lambda p: json.load(urllib.request.urlopen(base + p))  # noqa: E731
    try:
        assert get('/api/config') == {'title': 'Test round'}
        st = get('/api/state?a=AB')
        tasks = json.load(open(demo / 'build' / 'tasks.json'))['tasks']
        assert len(st['tasks']) == len(tasks) and st['status'] == {}
        assert all(set(t) == {'token', 'nm_per_px', 'w', 'h'} for t in st['tasks'])   # no repeat links
        assert get('/api/state?a=AB')['tasks'] == st['tasks']                          # order is stable
        t = st['tasks'][0]['token']
        body = json.dumps({'a': 'AB', 't': t, 'status': 'done', 'contours': [{'pts': [[1, 1], [5, 1], [5, 5]]}]}).encode()
        req = urllib.request.Request(base + '/api/save', body, {'Content-Type': 'application/json'})
        assert json.load(urllib.request.urlopen(req))['ok']
        assert get(f'/api/ann?a=AB&t={t}')['status'] == 'done'
        assert get('/api/state?a=AB')['status'][t] == {'status': 'done', 'n': 1}
        assert urllib.request.urlopen(f'{base}/img/{t}.png').headers['Content-Type'] == 'image/png'
        assert b'<title>' in urllib.request.urlopen(base + '/').read()
        assert urllib.request.urlopen(base).geturl().endswith('/bt/')                 # redirect to the slash
        bad = urllib.request.Request(base + '/api/save', json.dumps({'a': 'AB', 't': t, 'status': 'x', 'contours': []}).encode())
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + '/img/000000000000.png')
    finally:
        httpd.shutdown()
