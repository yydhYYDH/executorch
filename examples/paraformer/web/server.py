#!/usr/bin/env python3
"""Web bridge for the on-device Paraformer streaming runner.

Serves a small browser UI (upload a file or record from the microphone) and
relays audio to the runner started with --tcp_port. The runner protocol is
framed as one type byte, a little-endian uint32 payload length and the payload:

    client -> runner   0x01 START (empty)
                       0x02 raw signed 16-bit little-endian mono PCM at 16 kHz
                       0x03 END (empty)
    runner -> client   0x80 start status (uint32)
                       0x81 cumulative UTF-8 partial text
                       0x83 segment: start_ms, end_ms, speaker, text_length, text
                       0x84 diarization: count, then count x segment records (final)
                       0x82 final: status, elapsed_ms, audio_ms, text_length, text
                       0x7f error status (uint32)

The bridge exposes a synchronous upload endpoint and a streaming session made of
POSTed PCM chunks plus a Server-Sent Events stream of partial text and the final
result. Only the Python standard library is used.
"""

import argparse
import json
import os
import queue
import socket
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_AUDIO_BYTES = 256 * 1024 * 1024
PCM_CHUNK_BYTES = 4096
START, PCM, END = 0x01, 0x02, 0x03
START_OK, PARTIAL, SEGMENT, DIAR, FINAL, ERROR = 0x80, 0x81, 0x83, 0x84, 0x82, 0x7F
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
}


def _recv_exact(sock, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("runner closed the connection early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ConnectionError("client closed the request body early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock, frame_type, payload):
    sock.sendall(bytes([frame_type]) + struct.pack("<I", len(payload)) + payload)


def _recv_frame(sock):
    header = _recv_exact(sock, 5)
    length = struct.unpack("<I", header[1:5])[0]
    payload = _recv_exact(sock, length) if length else b""
    return header[0], payload


def _pcm_from_body(body):
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        offset = 12
        while offset + 8 <= len(body):
            chunk_id = body[offset:offset + 4]
            size = struct.unpack("<I", body[offset + 4:offset + 8])[0]
            if chunk_id == b"data":
                data = body[offset + 8:offset + 8 + size]
                return data[:len(data) - (len(data) % 2)]
            offset += 8 + size + (size & 1)
        raise ValueError("WAV has no data chunk")
    return body


def _parse_final(payload):
    status, elapsed_ms, audio_ms, text_length = struct.unpack("<IIII", payload[:16])
    text = payload[16:16 + text_length].decode("utf-8", "replace")
    rtf = elapsed_ms / audio_ms if audio_ms else 0.0
    return {"status": status, "text": text, "elapsed_ms": elapsed_ms, "audio_ms": audio_ms, "rtf": rtf}


def _parse_segment(payload):
    start_ms, end_ms, speaker, text_length = struct.unpack("<IIiI", payload[:16])
    text = payload[16:16 + text_length].decode("utf-8", "replace")
    return {"start_ms": start_ms, "end_ms": end_ms, "speaker": speaker, "text": text}


def _parse_diar(payload):
    count = struct.unpack("<I", payload[:4])[0]
    offset = 4
    segments = []
    for _ in range(count):
        start_ms, end_ms, speaker, text_length = struct.unpack("<IIiI", payload[offset:offset + 16])
        text = payload[offset + 16:offset + 16 + text_length].decode("utf-8", "replace")
        segments.append({"start_ms": start_ms, "end_ms": end_ms, "speaker": speaker, "text": text})
        offset += 16 + text_length
    return {"segments": segments}


class Session:
    def __init__(self, runner):
        self.sock = socket.create_connection((runner.host, runner.port), runner.timeout)
        self.sock.settimeout(runner.timeout)
        self._write_lock = threading.Lock()
        self.events = queue.Queue()
        self.audio_bytes = 0
        self.closed = False
        self.started = time.monotonic()
        try:
            with self._write_lock:
                _send_frame(self.sock, START, b"")
            frame_type, payload = _recv_frame(self.sock)
            if frame_type != START_OK or len(payload) < 4 or struct.unpack("<I", payload[:4])[0] != 0:
                raise RuntimeError("runner rejected the session start")
        except Exception:
            self.close()
            raise
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            while True:
                frame_type, payload = _recv_frame(self.sock)
                if frame_type == PARTIAL:
                    self.events.put(("partial", {"text": payload.decode("utf-8", "replace")}))
                elif frame_type == SEGMENT:
                    self.events.put(("segment", _parse_segment(payload)))
                elif frame_type == DIAR:
                    self.events.put(("diarization", _parse_diar(payload)))
                elif frame_type == FINAL:
                    self.events.put(("done", _parse_final(payload)))
                    return
                elif frame_type == ERROR:
                    status = struct.unpack("<I", payload[:4])[0] if len(payload) >= 4 else -1
                    self.events.put(("error", {"status": status}))
                    return
        except Exception as exc:  # noqa: BLE001 - forward transport failures to the UI
            self.events.put(("error", {"error": str(exc)}))
        finally:
            self.close()

    def feed(self, pcm):
        if self.closed or self.audio_bytes + len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("session closed or audio limit exceeded")
        with self._write_lock:
            _send_frame(self.sock, PCM, pcm)
        self.audio_bytes += len(pcm)

    def finish(self):
        with self._write_lock:
            _send_frame(self.sock, END, b"")

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.sock.close()
            except OSError:
                pass


class RunnerClient:
    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sessions = {}
        self._lock = threading.Lock()

    def probe(self):
        with socket.create_connection((self.host, self.port), min(self.timeout, 5.0)):
            return True

    def transcribe(self, body):
        pcm = _pcm_from_body(body)
        if not pcm or len(pcm) % 2:
            raise ValueError("audio must be nonempty 16-bit PCM")
        if len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("audio exceeds the 256 MiB runner limit")
        with socket.create_connection((self.host, self.port), self.timeout) as sock:
            sock.settimeout(self.timeout)
            _send_frame(sock, START, b"")
            frame_type, payload = _recv_frame(sock)
            if frame_type != START_OK or struct.unpack("<I", payload[:4])[0] != 0:
                raise RuntimeError("runner rejected the session start")
            for offset in range(0, len(pcm), PCM_CHUNK_BYTES):
                _send_frame(sock, PCM, pcm[offset:offset + PCM_CHUNK_BYTES])
                while True:
                    frame_type, payload = _recv_frame(sock)
                    if frame_type == SEGMENT:
                        continue
                    break
                if frame_type != PARTIAL:
                    status = struct.unpack("<I", payload[:4])[0] if len(payload) >= 4 else -1
                    raise RuntimeError("runner error %d" % status)
            _send_frame(sock, END, b"")
            while True:
                frame_type, payload = _recv_frame(sock)
                if frame_type == PARTIAL or frame_type == SEGMENT:
                    continue
                if frame_type == FINAL:
                    return _parse_final(payload)
                status = struct.unpack("<I", payload[:4])[0] if len(payload) >= 4 else -1
                raise RuntimeError("runner error %d" % status)

    def new_session(self):
        session = Session(self)
        with self._lock:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id):
        with self._lock:
            return self.sessions.get(session_id)

    def drop_session(self, session_id):
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if session:
            session.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "paraformer-web/0.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[web] " + fmt % args, flush=True)

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_AUDIO_BYTES:
            return None
        return _read_exact(self.rfile, length)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            try:
                reachable = self.server.runner.probe()
            except Exception:
                reachable = False
            self._json(200, {"ok": True, "runner": reachable})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "session"] and parts[3] == "events":
            self._events(parts[2])
            return
        self._serve_static(path)

    def do_HEAD(self):
        self._serve_static(self.path.split("?", 1)[0], head_only=True)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/transcribe":
            self._transcribe()
            return
        if path == "/api/session":
            self._new_session()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "session"]:
            if parts[3] == "audio":
                self._feed(parts[2])
                return
            if parts[3] == "end":
                self._finish(parts[2])
                return
        self._json(404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "session"]:
            self.server.runner.drop_session(parts[2])
            self._json(200, {"ok": True})
            return
        self._json(404, {"error": "not found"})

    def _transcribe(self):
        pcm = self._body()
        if pcm is None:
            self._json(400, {"error": "invalid or oversized body"})
            return
        try:
            result = self.server.runner.transcribe(pcm)
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": str(exc)})
            return
        self._json(200, result)

    def _new_session(self):
        try:
            session_id = self.server.runner.new_session()
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": str(exc)})
            return
        self._json(200, {"id": session_id})

    def _feed(self, session_id):
        session = self.server.runner.get_session(session_id)
        if session is None:
            self._json(404, {"error": "unknown session"})
            return
        body = self._body()
        if body is None:
            self._json(400, {"error": "invalid or oversized body"})
            return
        try:
            pcm = _pcm_from_body(body)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if not pcm or len(pcm) % 2:
            self._json(400, {"error": "audio must be nonempty 16-bit PCM"})
            return
        try:
            session.feed(pcm)
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": str(exc)})
            return
        self._json(200, {"ok": True, "audio_bytes": session.audio_bytes})

    def _finish(self, session_id):
        session = self.server.runner.get_session(session_id)
        if session is None:
            self._json(404, {"error": "unknown session"})
            return
        if session.audio_bytes == 0:
            self._json(400, {"error": "no audio received"})
            return
        try:
            session.finish()
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": str(exc)})
            return
        self._json(202, {"ok": True})

    def _events(self, session_id):
        session = self.server.runner.get_session(session_id)
        if session is None:
            self._json(404, {"error": "unknown session"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                try:
                    event, data = session.events.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(("event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))).encode("utf-8"))
                self.wfile.flush()
                if event in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            session.close()

    def _serve_static(self, path, head_only=False):
        if path in ("/", ""):
            path = "/index.html"
        root = self.server.static_dir
        target = os.path.normpath(os.path.join(root, path.lstrip("/")))
        if not target.startswith(root + os.sep) and target != root:
            self._json(403, {"error": "forbidden"})
            return
        if not os.path.isfile(target):
            self._json(404, {"error": "not found"})
            return
        with open(target, "rb") as handle:
            body = handle.read()
        ext = os.path.splitext(target)[1].lower()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-host", default="127.0.0.1", help="paraformer_runner TCP host")
    parser.add_argument("--runner-port", type=int, default=9000, help="paraformer_runner TCP port")
    parser.add_argument("--listen-host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--listen-port", type=int, default=8020, help="HTTP bind port")
    parser.add_argument("--static-dir", default=here, help="directory with index.html")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request runner timeout (s)")
    args = parser.parse_args()

    runner = RunnerClient(args.runner_host, args.runner_port, args.timeout)
    httpd = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    httpd.runner = runner
    httpd.static_dir = os.path.abspath(args.static_dir)
    print(
        "[web] serving %s on http://%s:%d, runner %s:%d"
        % (httpd.static_dir, args.listen_host, args.listen_port, args.runner_host, args.runner_port),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
