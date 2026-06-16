#!/usr/bin/env python3
"""Compatibility proxy for Yunwu's Anthropic-compatible endpoint.

Claude Code 2.1.x sends newer Anthropic beta headers and effort/thinking fields
that Yunwu's relay may accept with HTTP 200 but answer without content. This
proxy keeps Claude Code unchanged while stripping those unsupported fields before
forwarding requests to Yunwu.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import threading
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


DROP_HEADERS = {
    "accept-encoding",
    "anthropic-beta",
    "connection",
    "content-length",
    "host",
}

DROP_RESPONSE_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "transfer-encoding",
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _rewrite_body(raw: bytes) -> tuple[bytes, list[str], list[str]]:
    removed: list[str] = []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw, ["raw"], removed

    if isinstance(payload, dict):
        if "output_config" in payload:
            payload.pop("output_config", None)
            removed.append("output_config")
        if "thinking" in payload:
            payload.pop("thinking", None)
            removed.append("thinking")
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return raw, list(payload.keys()), removed
    return raw, [type(payload).__name__], removed


def make_handler(
    target_host: str,
    target_port: int,
    max_upstream_concurrency: int,
    max_429_retries: int,
    retry_base_delay: float,
) -> type[BaseHTTPRequestHandler]:
    upstream_slots = threading.BoundedSemaphore(max_upstream_concurrency)

    class YunwuCompatHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            self.close_connection = True
            started = time.time()
            content_len = int(self.headers.get("content-length", "0") or 0)
            raw = self.rfile.read(content_len)
            raw, keys, removed_fields = _rewrite_body(raw)

            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in DROP_HEADERS
            }
            headers["host"] = target_host
            headers["content-length"] = str(len(raw))

            path = self.path.split("?", 1)[0]
            total = 0
            status = 502
            slot_acquired = False
            try:
                attempt = 0
                upstream_slots.acquire()
                slot_acquired = True
                while True:
                    attempt += 1
                    conn = http.client.HTTPSConnection(target_host, target_port, timeout=600)
                    conn.request("POST", path, body=raw, headers=headers)
                    response = conn.getresponse()
                    status = response.status
                    if status != 429 or attempt > max_429_retries:
                        break
                    response_body = response.read()
                    upstream_slots.release()
                    slot_acquired = False
                    delay = retry_base_delay * min(30.0, 2 ** min(attempt - 1, 5))
                    delay *= random.uniform(0.75, 1.25)
                    print(
                        f"{_timestamp()} {path} upstream_429_retry attempt={attempt} "
                        f"bytes={len(response_body)} sleep={delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    upstream_slots.acquire()
                    slot_acquired = True

                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in DROP_RESPONSE_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()

                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    total += len(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as exc:  # pragma: no cover - operational fallback
                status = 502
                try:
                    self.send_response(502)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(str(exc).encode("utf-8", "replace"))
                except Exception:
                    pass
                print(f"{_timestamp()} proxy_error {type(exc).__name__}: {exc}", flush=True)
            finally:
                if slot_acquired:
                    upstream_slots.release()
                elapsed = time.time() - started
                print(
                    f"{_timestamp()} {path} status={status} bytes={total} "
                    f"elapsed={elapsed:.1f}s removed={','.join(removed_fields) or '-'} "
                    f"keys={','.join(keys)}",
                    flush=True,
                )

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return YunwuCompatHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18082)
    parser.add_argument("--target-base-url", default="https://yunwu.ai")
    parser.add_argument("--max-upstream-concurrency", type=int, default=1)
    parser.add_argument("--max-429-retries", type=int, default=12)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_upstream_concurrency < 1:
        parser.error("--max-upstream-concurrency must be >= 1")
    if args.max_429_retries < 0:
        parser.error("--max-429-retries must be >= 0")

    target = urlparse(args.target_base_url)
    if target.scheme != "https" or not target.hostname:
        parser.error("--target-base-url must be an https URL with a hostname")
    target_port = target.port or 443

    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port),
        make_handler(
            target.hostname,
            target_port,
            args.max_upstream_concurrency,
            args.max_429_retries,
            args.retry_base_delay,
        ),
    )
    print(
        f"{_timestamp()} listening on {args.listen_host}:{args.listen_port} "
        f"-> https://{target.hostname}:{target_port} "
        f"max_upstream_concurrency={args.max_upstream_concurrency} "
        f"max_429_retries={args.max_429_retries}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
