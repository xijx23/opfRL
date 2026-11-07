"""HTTP front-end for MatlabEnginePool (runs on the machine that has MATLAB).

POST /execute  {"programs": ["...", ...]}  ->  {"results": [ExecResult, ...]}
GET  /health                               ->  {"status": "ok", "engines": N}
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .matlab_pool import MatlabEnginePool


def make_handler(pool: MatlabEnginePool):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "engines": pool.cfg.num_engines})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/execute":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                programs = json.loads(self.rfile.read(length))["programs"]
            except (ValueError, KeyError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            results = pool.execute(programs)
            self._send(200, {"results": [r.to_dict() for r in results]})

        def log_message(self, fmt, *args):  # keep the console readable
            pass

    return Handler


def serve(pool: MatlabEnginePool, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(pool))
    print(f"[verifier] {pool.cfg.num_engines} MATLAB engine(s) ready on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.close()
