#!/usr/bin/env python3
"""
automatch_server -- run automatch jobs on a GPU server and drive them with curl.

The jobs are the same JSON job files the GUI and `automatch_job.py` use, run
by the same code (`automatch_job.py run` as a subprocess); this only adds an
HTTP front end. Python standard library only.

    python automatch_server.py --port 8765 --jobs-dir /data/automatch_jobs [--token SECRET]

It listens on 127.0.0.1 unless --host is given; from a workstation use an SSH
tunnel (ssh -L 8765:localhost:8765 user@server) or --host 0.0.0.0 with --token.
Jobs run one at a time on the GPU (queue); --parallel N runs N at once.

Endpoints (JSON in and out; paths in job files are paths ON THE SERVER):
    GET  /health
    GET  /detectors                         catalogue (detectors, matchers, settings)
    POST /preflight          <job json>     validate, no matching
    POST /weights            <job json>     which weight files the job needs / has
    POST /jobs               <job json>     queue a run -> {"id": ...}
    GET  /jobs                              all jobs and their state
    GET  /jobs/<id>                         state, progress, results, truth ranking
    GET  /jobs/<id>/log?tail=200            last lines of the run log
    GET  /jobs/<id>/files                   files in the job's output folder
    GET  /jobs/<id>/files/<path>            download one (e.g. TRUTH_BY_DETECTOR_MATCHER.csv)
    POST /jobs/<id>/stop
    POST /compare   {"output_dir": ..., "truth_csv": ..., "radius_km": 5}

Send the token, when set, as   -H "Authorization: Bearer SECRET".
"""

import argparse
import collections
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_SCRIPT = os.path.join(HERE, 'automatch_job.py')
PREFIX = '@@AUTOMATCH '


class Job:
    def __init__(self, jid: str, job: Dict, folder: str):
        self.id = jid
        self.job = job
        self.folder = folder
        self.state = 'queued'          # queued | running | done | failed | stopped
        self.created = time.time()
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.returncode: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.progress: Dict = {}
        self.results: List[Dict] = []
        self.truth: Optional[Dict] = None
        self.done: Optional[Dict] = None
        self.errors: List[str] = []
        self.log_path = os.path.join(folder, 'server_run.log')

    def summary(self, full: bool = False) -> Dict:
        out = {'id': self.id, 'state': self.state, 'output_dir': self.job.get('output_dir'),
               'detectors': self.job.get('detectors'), 'returncode': self.returncode,
               'created': _iso(self.created), 'started': _iso(self.started), 'ended': _iso(self.ended),
               'minutes': round(((self.ended or time.time()) - self.started) / 60.0, 1) if self.started else None,
               'progress': self.progress, 'errors': self.errors}
        if full:
            out.update({'results': self.results, 'truth': self.truth, 'done': self.done, 'job': self.job})
        return out


def _iso(t):
    return datetime.datetime.fromtimestamp(t).isoformat(timespec='seconds') if t else None


class Server:
    def __init__(self, jobs_dir: str, parallel: int = 1, python: str = sys.executable):
        self.jobs_dir = os.path.abspath(jobs_dir)
        os.makedirs(self.jobs_dir, exist_ok=True)
        self.parallel = max(1, parallel)
        self.python = python
        self.jobs: Dict[str, Job] = collections.OrderedDict()
        self.lock = threading.Lock()
        threading.Thread(target=self._scheduler, daemon=True).start()

    # ── jobs ────────────────────────────────────────────────────────────────
    def submit(self, job: Dict) -> Job:
        if not job.get('output_dir'):
            raise ValueError('job needs output_dir (a folder on the server)')
        jid = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        folder = os.path.join(self.jobs_dir, jid)
        os.makedirs(folder)
        with open(os.path.join(folder, 'job.json'), 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        j = Job(jid, job, folder)
        with self.lock:
            self.jobs[jid] = j
        return j

    def _scheduler(self):
        while True:
            with self.lock:
                running = sum(1 for j in self.jobs.values() if j.state == 'running')
                nxt = next((j for j in self.jobs.values() if j.state == 'queued'), None)
                if nxt is not None and running < self.parallel:
                    nxt.state = 'running'
                    nxt.started = time.time()
                else:
                    nxt = None
            if nxt is not None:
                threading.Thread(target=self._run, args=(nxt,), daemon=True).start()
            time.sleep(1.0)

    def _run(self, j: Job):
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        with open(j.log_path, 'w', encoding='utf-8') as log:
            j.proc = subprocess.Popen([self.python, JOB_SCRIPT, 'run', os.path.join(j.folder, 'job.json')],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                      text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in j.proc.stdout:
                if line.startswith(PREFIX):
                    try:
                        self._event(j, json.loads(line[len(PREFIX):]))
                    except ValueError:
                        pass
                    continue
                log.write(line)
                log.flush()
            j.returncode = j.proc.wait()
        j.ended = time.time()
        if j.state == 'running':
            j.state = 'done' if j.returncode == 0 else 'failed'

    @staticmethod
    def _event(j: Job, ev: Dict):
        kind = ev.get('event')
        if kind in ('job', 'sweep', 'detector', 'stage', 'window', 'preprocess'):
            j.progress[kind] = ev
        elif kind == 'result':
            j.results.append(ev)
        elif kind == 'truth_result':
            for r in j.results:
                if all(r.get(k) == ev.get(k) for k in ('sweep', 'channel', 'detector')):
                    r.update({'truth_rmse_m': ev.get('truth_rmse_m'), 'truth_reached': ev.get('truth_reached')})
        elif kind == 'truth':
            j.truth = ev
        elif kind == 'done':
            j.done = ev
        elif kind == 'failed':
            j.errors += ev.get('errors', [])

    def stop(self, j: Job):
        if j.state == 'queued':
            j.state = 'stopped'
        elif j.state == 'running' and j.proc is not None:
            j.state = 'stopped'
            j.proc.terminate()
            try:
                j.proc.wait(10)
            except subprocess.TimeoutExpired:
                j.proc.kill()

    # ── quick commands (run to completion, return their event) ───────────────
    def command(self, args: List[str], event: str, timeout: float = 3600) -> Dict:
        res = subprocess.run([self.python, JOB_SCRIPT] + args, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=timeout,
                             env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        for line in res.stdout.splitlines():
            if line.startswith(PREFIX):
                ev = json.loads(line[len(PREFIX):])
                if ev.get('event') == event:
                    return {**ev, 'returncode': res.returncode}
        return {'event': event, 'returncode': res.returncode,
                'output': (res.stdout + res.stderr)[-4000:]}

    def job_file(self, job: Dict) -> str:
        path = os.path.join(self.jobs_dir, f'_tmp_{uuid.uuid4().hex}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f)
        return path


def make_handler(srv: Server, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'automatch/1'

        def log_message(self, fmt, *args):
            sys.stderr.write(f'[{self.log_date_time_string()}] {self.address_string()} {fmt % args}\n')

        def _send(self, code: int, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, indent=2, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self) -> bool:
            if not token or self.headers.get('Authorization', '') == f'Bearer {token}':
                return True
            self._send(401, {'error': 'missing or wrong token (Authorization: Bearer ...)'})
            return False

        def _body(self) -> Dict:
            n = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(n) if n else b''
            return json.loads(raw.decode('utf-8-sig')) if raw.strip() else {}

        def _job(self, jid: str) -> Optional[Job]:
            j = srv.jobs.get(jid)
            if j is None:
                self._send(404, {'error': f'no job {jid}'})
            return j

        def do_GET(self):
            if urlparse(self.path).path.strip('/') == 'health':
                return self._send(200, {'ok': True, 'jobs': len(srv.jobs), 'parallel': srv.parallel})
            if not self._auth():
                return
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip('/').split('/') if p]
            q = parse_qs(u.query)
            try:
                if parts == ['health']:
                    return self._send(200, {'ok': True, 'jobs': len(srv.jobs), 'parallel': srv.parallel})
                if parts == ['detectors']:
                    return self._send(200, srv.command(['detectors'], 'detectors', 600))
                if parts == ['jobs']:
                    return self._send(200, [j.summary() for j in srv.jobs.values()])
                if len(parts) >= 2 and parts[0] == 'jobs':
                    j = self._job(parts[1])
                    if j is None:
                        return
                    if len(parts) == 2:
                        return self._send(200, j.summary(full=True))
                    if parts[2] == 'log':
                        n = int((q.get('tail') or ['200'])[0])
                        lines = open(j.log_path, encoding='utf-8', errors='replace').read().splitlines() \
                            if os.path.exists(j.log_path) else []
                        return self._send(200, ('\n'.join(lines[-n:]) + '\n').encode(), 'text/plain; charset=utf-8')
                    if parts[2] == 'files':
                        root = os.path.realpath(j.job['output_dir'])
                        if len(parts) == 3:
                            out = []
                            for dp, _, fs in os.walk(root):
                                depth = os.path.relpath(dp, root).count(os.sep)
                                if depth > 1:
                                    continue
                                out += [os.path.relpath(os.path.join(dp, f), root) for f in sorted(fs)]
                            return self._send(200, out)
                        path = os.path.realpath(os.path.join(root, *parts[3:]))
                        if not path.startswith(root + os.sep) or not os.path.isfile(path):
                            return self._send(404, {'error': 'no such file in the output folder'})
                        ctype = 'text/csv' if path.endswith('.csv') else 'application/octet-stream'
                        with open(path, 'rb') as f:
                            return self._send(200, f.read(), ctype)
                self._send(404, {'error': 'unknown path', 'see': 'python automatch_server.py -h'})
            except Exception as e:
                self._send(500, {'error': f'{type(e).__name__}: {e}'})

        def do_POST(self):
            if not self._auth():
                return
            parts = [unquote(p) for p in urlparse(self.path).path.strip('/').split('/') if p]
            try:
                body = self._body()
                if parts == ['jobs']:
                    j = srv.submit(body)
                    return self._send(201, {'id': j.id, 'state': j.state, 'status_url': f'/jobs/{j.id}'})
                if parts == ['preflight']:
                    path = srv.job_file(body)
                    try:
                        return self._send(200, srv.command(['preflight', path], 'preflight', 1800))
                    finally:
                        os.remove(path)
                if parts == ['weights']:
                    path = srv.job_file(body)
                    try:
                        return self._send(200, srv.command(['weights', path], 'weights', 1800))
                    finally:
                        os.remove(path)
                if parts == ['compare']:
                    args = ['compare', body['output_dir'], '--truth', body['truth_csv'],
                            '--radius-km', str(body.get('radius_km', 5))]
                    res = subprocess.run([srv.python, JOB_SCRIPT] + args, capture_output=True, text=True,
                                         encoding='utf-8', errors='replace')
                    return self._send(200, {'returncode': res.returncode, 'output': res.stdout[-8000:],
                                            'files': [os.path.join(body['output_dir'], f'{k}.csv') for k in
                                                      ('TRUTH_BY_DETECTOR_MATCHER', 'TRUTH_RANKING',
                                                       'TRUTH_POINTS')]})
                if len(parts) == 3 and parts[0] == 'jobs' and parts[2] == 'stop':
                    j = self._job(parts[1])
                    if j is not None:
                        srv.stop(j)
                        self._send(200, j.summary())
                    return
                self._send(404, {'error': 'unknown path'})
            except (ValueError, KeyError) as e:
                self._send(400, {'error': f'{type(e).__name__}: {e}'})
            except Exception as e:
                self._send(500, {'error': f'{type(e).__name__}: {e}'})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='HTTP front end for automatch jobs (drive it with curl).')
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to accept other machines (set --token)')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--jobs-dir', default=os.path.join(os.getcwd(), 'automatch_jobs'))
    ap.add_argument('--parallel', type=int, default=1, help='jobs running at once (default 1: one GPU)')
    ap.add_argument('--token', default=os.environ.get('AUTOMATCH_TOKEN', ''),
                    help='require "Authorization: Bearer TOKEN" (or env AUTOMATCH_TOKEN)')
    a = ap.parse_args(argv)
    if a.host not in ('127.0.0.1', 'localhost', '::1') and not a.token:
        print('Refusing to listen beyond localhost without --token.')
        return 2
    srv = Server(a.jobs_dir, a.parallel)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv, a.token))
    print(f'automatch server on http://{a.host}:{a.port}  jobs in {srv.jobs_dir}  '
          f'(token {"on" if a.token else "off"}, {srv.parallel} job(s) at a time)')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
