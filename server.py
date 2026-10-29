"""BlindTrace server: blinded particle tracing with Apple Pencil (or any touch/pen device).

Serves the single-page tracing app, the blinded task images and stores the tracings.
Images carry only opaque tokens; the unblinding key made by prepare.py never goes on
the server. The server has no login: run it on a trusted network, or behind a reverse
proxy with HTTP Basic Auth and HTTPS (see deploy/). Python standard library only.

Layout
  DATA/tasks.json          {title, tasks: [{token, nm_per_px, w, h, after}]}, 'after' =
                           token of the original for a hidden repeat (never sent to the app)
  DATA/images/<token>.png
  WEB/index.html
  STATE/orders/<annotator>.json      per-annotator task order (fixed at first visit)
  STATE/ann/<annotator>/<token>.json one file per annotator and image
  STATE/log.jsonl                    audit log of every save

Routes (below PREFIX, default none)
  GET  /                  app
  GET  /img/<token>.png   image
  GET  /api/config        {title}
  GET  /api/state?a=AB    task order and per-task status of annotator AB
  GET  /api/ann?a=AB&t=…  saved tracing of one task
  POST /api/save          {a, t, status, contours, …}

Usage: python server.py --data build --state state [--host 0.0.0.0 --port 8170 --prefix /blindtrace]
Every option can also be set as an environment variable (BLINDTRACE_DATA, …).
"""
import argparse
import json
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_GAP = 10           # a repeat comes at least this many tasks after its original
MAX_BODY = 2_000_000
NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,16}$')
TOKEN_RE = re.compile(r'^[0-9a-f]{12}$')
STATUSES = {'draft', 'done', 'excluded'}

LOCK = threading.Lock()
DATA = WEB = STATE = PREFIX = TITLE = None
TASKS = {}


def make_order(name):
    """Random task order per annotator (seeded by the name, so it is stable); each hidden
    repeat is placed at least MIN_GAP tasks after its original."""
    rng = random.Random(f'blindtrace:{name}')
    orig = [t for t in TASKS.values() if not t['after']]
    reps = [t for t in TASKS.values() if t['after']]
    order = [t['token'] for t in orig]
    rng.shuffle(order)
    for r in sorted(reps, key=lambda t: order.index(t['after'])):
        lo = order.index(r['after']) + MIN_GAP
        pos = rng.randint(lo, len(order)) if lo <= len(order) else len(order)
        order.insert(pos, r['token'])
    return order


def get_order(name):
    p = os.path.join(STATE, 'orders', f'{name}.json')
    with LOCK:
        if os.path.exists(p):
            return json.load(open(p))
        order = make_order(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        write_json(p, order)
        return order


def write_json(path, obj):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def ann_path(name, token):
    return os.path.join(STATE, 'ann', name, f'{token}.json')


def valid_contours(cs):
    if not isinstance(cs, list) or len(cs) > 1000:
        return False
    for c in cs:
        if not isinstance(c, dict):
            return False
        for k in ('pts', 'drawn', 'base'):
            p = c.get(k, [])
            if not isinstance(p, list) or len(p) > 600:
                return False
            if any(not (isinstance(q, list) and len(q) == 2 and all(isinstance(v, (int, float)) for v in q)) for q in p):
                return False
    return True


class H(BaseHTTPRequestHandler):
    server_version = 'blindtrace'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json', cache='no-store'):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', cache)
        self.end_headers()
        self.wfile.write(body)

    def _name(self, q):
        a = (q.get('a') or [''])[0]
        return a if NAME_RE.match(a) else None

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if PREFIX and path == PREFIX:     # the app resolves its API relative to the page: needs the slash
            self.send_response(301)
            self.send_header('Location', PREFIX + '/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path in (PREFIX + '/', PREFIX + '/index.html'):
            return self._send(200, open(os.path.join(WEB, 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
        m = re.fullmatch(re.escape(PREFIX) + r'/img/([0-9a-f]{12})\.png', path)
        if m:
            if m.group(1) not in TASKS:
                return self._send(404, {'error': 'unknown image'})
            return self._send(200, open(os.path.join(DATA, 'images', f'{m.group(1)}.png'), 'rb').read(), 'image/png',
                              'private, max-age=31536000, immutable')
        if path == PREFIX + '/api/config':
            return self._send(200, {'title': TITLE})
        if path == PREFIX + '/api/state':
            name = self._name(q)
            if not name:
                return self._send(400, {'error': 'initials: letters, digits, _ or -'})
            order = get_order(name)
            status = {}
            for t in order:
                p = ann_path(name, t)
                if os.path.exists(p):
                    j = json.load(open(p))
                    status[t] = {'status': j.get('status'), 'n': len(j.get('contours', []))}
            tasks = [{'token': t, 'nm_per_px': TASKS[t]['nm_per_px'], 'w': TASKS[t]['w'], 'h': TASKS[t]['h']}
                     for t in order]
            return self._send(200, {'tasks': tasks, 'status': status})
        if path == PREFIX + '/api/ann':
            name, t = self._name(q), (q.get('t') or [''])[0]
            if not name or not TOKEN_RE.match(t) or t not in TASKS:
                return self._send(400, {'error': 'bad request'})
            p = ann_path(name, t)
            return self._send(200, json.load(open(p)) if os.path.exists(p) else {})
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        if urlparse(self.path).path != PREFIX + '/api/save':
            return self._send(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > MAX_BODY:
            return self._send(413, {'error': 'body size'})
        try:
            d = json.loads(self.rfile.read(n))
        except ValueError:
            return self._send(400, {'error': 'json'})
        name, t = d.get('a', ''), d.get('t', '')
        if not NAME_RE.match(str(name)) or not TOKEN_RE.match(str(t)) or t not in TASKS:
            return self._send(400, {'error': 'annotator or task'})
        if d.get('status') not in STATUSES or not valid_contours(d.get('contours')):
            return self._send(400, {'error': 'status or contours'})
        rec = {k: d.get(k) for k in ('status', 'contours', 'reason', 't_open', 't_saved', 'view', 'client')}
        rec.update({'annotator': name, 'token': t, 'server_time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        os.makedirs(os.path.join(STATE, 'ann', name), exist_ok=True)
        with LOCK:
            write_json(ann_path(name, t), rec)
            with open(os.path.join(STATE, 'log.jsonl'), 'a') as f:
                f.write(json.dumps({'t': rec['server_time'], 'a': name, 'task': t, 'status': rec['status'],
                                    'n': len(rec['contours'] or [])}) + '\n')
        return self._send(200, {'ok': True, 'server_time': rec['server_time']})


def setup(data, state, web=os.path.join(HERE, 'web'), prefix=''):
    """Load the task set; also used by the tests to run the server in-process."""
    global DATA, WEB, STATE, PREFIX, TITLE, TASKS
    DATA, WEB, STATE, PREFIX = data, web, state, prefix.rstrip('/')
    j = json.load(open(os.path.join(DATA, 'tasks.json')))
    TITLE = j.get('title') or 'Particle tracing'
    TASKS = {t['token']: t for t in j['tasks']}
    os.makedirs(STATE, exist_ok=True)


def main():
    env = lambda k, d: os.environ.get(f'BLINDTRACE_{k}', d)  # noqa: E731
    ap = argparse.ArgumentParser(description='Serve a prepared BlindTrace task set.')
    ap.add_argument('--data', default=env('DATA', 'build'), help='folder made by prepare.py (tasks.json, images/)')
    ap.add_argument('--state', default=env('STATE', 'state'), help='where tracings are stored')
    ap.add_argument('--web', default=env('WEB', os.path.join(HERE, 'web')))
    ap.add_argument('--host', default=env('HOST', '127.0.0.1'), help='0.0.0.0 to reach it from a tablet on the network')
    ap.add_argument('--port', type=int, default=int(env('PORT', '8170')))
    ap.add_argument('--prefix', default=env('PREFIX', ''), help='URL path when behind a proxy, e.g. /blindtrace')
    a = ap.parse_args()
    setup(a.data, a.state, a.web, a.prefix)
    print(f'{len(TASKS)} tasks, "{TITLE}": http://{a.host}:{a.port}{PREFIX}/  (tracings -> {os.path.abspath(STATE)})', flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()
