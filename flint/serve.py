"""Serving a Flint checkpoint over HTTP.

The endpoint is TypeSafe's System One shape (``POST /v1/systemone``), so a client
written against that API can be pointed at a local Flint by changing a base URL.
The choice is deliberate: it makes the format someone else's contract rather than
ours, and it means a suite can be replayed against a running server verbatim
instead of through a second implementation that drifts from the first.

Nothing here decides whether an answer is good enough to use. The response
carries the confidence; ``--threshold`` additionally reports a ``release`` flag
so a caller can see what the configured operating point would do, but refusing to
answer is the caller's decision — only the caller knows what a wrong answer
costs.

Standard library only. A local single-user model server does not need a web
framework, and a dependency-free one is easier to reason about than a small one.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["FlintService", "make_handler", "main"]

MAX_BODY = 8 << 20  # 8 MiB: a state with a few thousand options fits, a mistake does not.


class FlintService:
    """Holds the loaded checkpoint and answers requests against it."""

    def __init__(self, checkpoint, threshold: float | None = None):
        self.checkpoint = checkpoint
        self.threshold = threshold
        self._lock = threading.Lock()

    def decide(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("the request body must be a JSON object")
        state = payload.get("state")
        questions = payload.get("questions")
        if state is None:
            raise ValueError("the request has no `state`")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("the request has no `questions`")

        record = {"state": state, "questions": questions}
        # One at a time: a decision model is asked in the middle of a keystroke
        # and the encoder is not re-entrant across threads sharing a device.
        with self._lock:
            answers = self.checkpoint.answer([record])[0]

        out = {}
        for key, answer in answers.items():
            entry = {
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer["confidence"],
            }
            if self.threshold is not None:
                entry["release"] = bool(answer["confidence"] >= self.threshold)
            out[key] = entry
        return out


def make_handler(service: FlintService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "flint"

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - the base class names these
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                self._send(200, {
                    "status": "ok",
                    "temperature": service.checkpoint.temperature,
                    "temperatureByOptions": service.checkpoint.temperature_by_options,
                    "threshold": service.threshold,
                })
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length") or 0)
            if length <= 0:
                self._send(400, {"error": "empty request body"})
                return
            if length > MAX_BODY:
                self._send(413, {"error": "request body over %d bytes" % MAX_BODY})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": "body is not valid JSON: %s" % exc})
                return
            try:
                self._send(200, service.decide(payload))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            # The default handler writes to stderr on every request, which a
            # per-keystroke caller turns into a flood.
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=None,
                    help="report a `release` flag below this confidence; omit to report only")
    args = ap.parse_args()

    from .inference import FlintCheckpoint

    checkpoint = FlintCheckpoint.load(args.checkpoint, device=args.device)
    service = FlintService(checkpoint, threshold=args.threshold)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print("serving %s on http://%s:%d/v1/systemone" % (args.checkpoint, args.host, args.port),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
