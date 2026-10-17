#!/usr/bin/env python3
"""lora-shim: one llama-server, several model ids (vLLM --lora-modules behaviour for llama.cpp).

llama-server loads the adapters with --lora ... --lora-init-without-apply (scale 0). This proxy sits on the public chat port:
/v1/models lists the base plus one id per LoRA, and a chat/completion request whose "model" is a LoRA id is forwarded as the base
model with {"lora": [{"id": N, "scale": 1.0}]}, so the adapter applies to that request only. A request for the base sends every
adapter at scale 0 explicitly (an empty list means "server defaults", which keep every --lora at scale 1). Everything else
(streaming, health, metrics, embeddings, /props, /slots) passes through unchanged. One engine, shared weights and slots; the only
extra memory is the adapter file itself. Standard library only, Python 3.9+.

usage: lora-shim.py --listen PORT [--listen-host H] --upstream PORT --base ALIAS [--lora ALIAS ...] [--lora-slot N] [-- llama-server ARGS...]
       (--lora order = adapter id order = the --lora order given to llama-server)

--lora-slot N pins every LoRA request to slot N, only when the request carries no id_slot of its own. llama-server clears a slot's
cache whenever the adapter set on that slot changes, so keeping the adapter on one slot and the base on the others preserves both
prefix caches. With a command after "--" the shim starts llama-server as a child and exits when it exits; without one it only proxies.

Chaining behind llama.cpp-ssd-cache-shim: put that shim on the public port and this one on a loopback port behind it. The contract
kept here: every request header is forwarded; the body is forwarded with only "model" and "lora" rewritten, so an id_slot set
upstream survives; --lora-slot yields to an existing id_slot; every other path is proxied unchanged. Give the cache shim a matching
MODEL_SLOTS entry for the adapter id. See README.md.
"""
import argparse, http.client, json, os, signal, subprocess, sys, threading, time
DUMP_DIR = os.environ.get("SHIM_DUMP_DIR")  # optional: save every rewritten chat request body here (diagnostics; off unless set)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REWRITE_PATHS = {"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/completion"}
MODELS_PATHS = {"/v1/models", "/models"}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate", "content-length", "host"}

def log(msg): print(f"shim: {msg}", flush=True)

class Shim(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg = None  # set in main
    def log_message(self, *a): pass

    def _send(self, status, headers, body=None):
        self.send_response(status)
        for k, v in headers: self.send_header(k, v)
        if body is not None: self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close"); self.end_headers()
        if body: self.wfile.write(body)

    def _upstream(self, method, path, body, headers):
        conn = http.client.HTTPConnection("127.0.0.1", self.cfg.upstream, timeout=3600)
        conn.request(method, path, body=body, headers=headers)
        return conn, conn.getresponse()

    def proxy(self):
        cfg = self.cfg; path = self.path; method = self.command
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        fwd = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        requested = None
        if method == "POST" and path.split("?")[0] in REWRITE_PATHS and body:
            try:
                data = json.loads(body); requested = data.get("model")
                if requested in cfg.lora_ids:
                    data["model"] = cfg.base; data["lora"] = [{"id": cfg.lora_ids[requested], "scale": 1.0}]
                    if cfg.lora_slot is not None and "id_slot" not in data:
                        # Pin every LoRA request to one slot so its KV cache is never cleared by an adapter switch
                        # and base traffic (which stays unpinned) keeps the other slots for batching.
                        data["id_slot"] = cfg.lora_slot
                else:
                    # base (or unknown id): every adapter explicitly at 0. An EMPTY list means "server defaults", and the
                    # defaults keep each --lora at scale 1 even with --lora-init-without-apply (server-context.cpp, 2026-10-16).
                    data["lora"] = [{"id": i, "scale": 0.0} for i in range(len(cfg.lora_order))]
                body = json.dumps(data).encode()
                if DUMP_DIR:
                    os.makedirs(DUMP_DIR, exist_ok=True)
                    with open(os.path.join(DUMP_DIR, time.strftime("%H%M%S") + f"_{requested}.json"), "wb") as fh: fh.write(body)
            except (ValueError, AttributeError):
                requested = None
        try:
            conn, resp = self._upstream(method, path, body, fwd)
        except OSError as e:
            self._send(503, [("Content-Type", "application/json")], json.dumps({"error": {"message": f"upstream llama-server not ready: {e}", "type": "unavailable"}}).encode()); return
        try:
            if method == "GET" and path.split("?")[0] in MODELS_PATHS and resp.status == 200:
                data = json.loads(resp.read()); items = data.get("data", [])
                base = next((m for m in items if m.get("id") == cfg.base), items[0] if items else {"object": "model", "created": int(time.time())})
                for alias in cfg.lora_order:
                    if not any(m.get("id") == alias for m in items):
                        m = dict(base); m["id"] = alias; m["owned_by"] = f"lora on {cfg.base}"; items.append(m)
                data["data"] = items
                self._send(200, [("Content-Type", "application/json")], json.dumps(data).encode()); return
            hdrs = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP]
            rewrite = requested in cfg.lora_ids
            if rewrite and "text/event-stream" not in (resp.getheader("Content-Type") or "") and resp.getheader("Content-Length"):
                out = resp.read()
                try:
                    d = json.loads(out); d["model"] = requested; out = json.dumps(d).encode()
                except ValueError: pass
                self._send(resp.status, hdrs, out); return
            self.send_response(resp.status)
            for k, v in hdrs: self.send_header(k, v)
            self.send_header("Connection", "close"); self.end_headers()
            needle = json.dumps(cfg.base).encode(); repl = json.dumps(requested).encode() if rewrite else None
            while True:
                chunk = resp.read1(65536)
                if not chunk: break
                if repl: chunk = chunk.replace(b'"model":' + needle, b'"model":' + repl)
                self.wfile.write(chunk); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = proxy

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, required=True); ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--upstream", type=int, required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--lora", action="append", default=[], help="model id for adapter N (repeat; order = --lora order of llama-server)")
    ap.add_argument("--lora-slot", type=int, default=None, help="llama-server slot id to pin every LoRA request to (default: no pinning)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- llama-server and its arguments")
    cfg = ap.parse_args()
    if cfg.cmd and cfg.cmd[0] == "--": cfg.cmd = cfg.cmd[1:]
    cfg.lora_order = cfg.lora; cfg.lora_ids = {a: i for i, a in enumerate(cfg.lora)}
    Shim.cfg = cfg

    child = None
    if cfg.cmd:
        log(f"starting llama-server on 127.0.0.1:{cfg.upstream}: {' '.join(cfg.cmd)}")
        child = subprocess.Popen(cfg.cmd)
    def stop(signum, frame):
        if child is not None:
            log(f"signal {signum}: stopping llama-server pid {child.pid}")
            child.terminate()
            try: child.wait(15)
            except subprocess.TimeoutExpired: child.kill()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    srv = ThreadingHTTPServer((cfg.listen_host, cfg.listen), Shim); srv.daemon_threads = True
    log(f"listening on {cfg.listen_host}:{cfg.listen} -> 127.0.0.1:{cfg.upstream}; model ids: {cfg.base} + {cfg.lora_order}; lora slot: {cfg.lora_slot}")
    if child is None:
        srv.serve_forever(); return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rc = child.wait()
    log(f"llama-server exited with {rc}; shim exiting")
    sys.exit(rc)

if __name__ == "__main__": main()
