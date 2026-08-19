"""Live browser visualization of a swarm run — optional, zero dependencies.

A tiny threaded HTTP server (stdlib only) serves a single-page graph UI
and streams the task tree over Server-Sent Events:

- ``/``        the page (Cytoscape.js graph: tasks, hierarchy, dependency
               edges, statuses, per-node details);
- ``/events``  SSE stream: full snapshot on connect, then only the nodes
               that changed (delta), throttled; heartbeats in between.

Raw worker outputs (``result_full``) never leave the process — the stream
carries truncated descriptions and summaries only, mirroring the swarm's
own context-isolation rule.

Usage (Python)::

    from myriapod.viz import serve

    with serve(swarm) as url:          # opens the browser by default
        result = swarm.run("...")      # watch it live

Usage (CLI)::

    myriapod ask "..." --viz
    myriapod bench -n 1000 -c 500 --viz    # 1000 agents, no API keys
"""

from __future__ import annotations

import json
import logging
import threading
import time
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator, Optional

from myriapod.core.task_tree import TaskTree
from myriapod.viz.page import PAGE_HTML

logger = logging.getLogger("myriapod.viz")

#: Task briefs run to several sentences, and the details panel is meant to be
#: read — but the wire still stays bounded, and never carries ``result_full``.
_DESC_MAX = 600
_TEXT_MAX = 400


def _node_payload(nd: dict[str, Any]) -> dict[str, Any]:
    """Shrink one serialized TaskNode for the wire. Never includes result_full."""
    return {
        "id": nd["id"],
        "parent": nd.get("parent_id"),
        "deps": nd.get("depends_on") or [],
        "desc": (nd.get("description") or "")[:_DESC_MAX],
        "status": nd["status"],
        "turn": nd.get("turn", 0),
        "worker": nd.get("worker"),
        "attempts": nd.get("attempts", 0),
        "cost": round(nd.get("cost", 0.0), 6),
        "itok": nd.get("input_tokens", 0),
        "otok": nd.get("output_tokens", 0),
        "dur": round(nd.get("duration", 0.0), 2),
        "summary": (nd.get("result_summary") or "")[:_TEXT_MAX],
        "error": (nd.get("error") or "")[:_TEXT_MAX],
    }


def _fingerprint(p: dict[str, Any]) -> tuple:
    return (p["status"], p["attempts"], p["cost"], p["summary"], p["error"], p["worker"])


class _Handler(BaseHTTPRequestHandler):
    server: "VizServer"  # type: ignore[assignment]

    def log_message(self, *args: Any) -> None:  # silence stdlib access logs
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/events":
            self._stream_events()
            return
        self.send_response(404)
        self.end_headers()

    # -- SSE ----------------------------------------------------------- #

    def _send_event(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        sent: dict[str, tuple] = {}
        sent_run_id: str | None = None
        ended_sent = False
        try:
            while True:
                tree = self.server.tree_getter()
                if tree is None:
                    self._send_event({"type": "waiting"})
                else:
                    snapshot = tree.to_dict()
                    fresh = snapshot["run_id"] != sent_run_id
                    if fresh:
                        sent, sent_run_id = {}, snapshot["run_id"]
                        ended_sent = False
                    changed: list[dict[str, Any]] = []
                    for nid, nd in snapshot["nodes"].items():
                        payload = _node_payload(nd)
                        fp = _fingerprint(payload)
                        if sent.get(nid) != fp:
                            sent[nid] = fp
                            changed.append(payload)
                    changed.sort(key=lambda p: (len(p["id"]), p["id"]))
                    if fresh or changed:
                        self._send_event(
                            {
                                "type": "snapshot" if fresh else "delta",
                                "run_id": sent_run_id,
                                "goal": tree.goal[:200],
                                "nodes": changed,
                                # No turn array: the graph holds agents only.
                                # A task carries the turn that planned it
                                # (``_node_payload``), which is all the page
                                # needs to lay the waves out; the planner's own
                                # spend is booked on the root node.
                                "stats": tree.stats(),
                            }
                        )
                    elif self.server.ended and not ended_sent:
                        self._send_event({"type": "end", "run_id": sent_run_id})
                        ended_sent = True
                    else:
                        self.wfile.write(b": hb\n\n")
                        self.wfile.flush()
                time.sleep(self.server.interval)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return  # client went away


class VizServer(ThreadingHTTPServer):
    """Serve the live graph for whatever ``tree_getter`` currently returns.

    ``tree_getter`` is typically ``lambda: swarm.tree`` — the scheduler
    publishes its live tree there at the start of every run.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Swallow the traceback a closing browser tab would otherwise print.

        A client that goes away mid-stream surfaces as ConnectionResetError /
        BrokenPipeError from the socket read, which ``socketserver`` dumps to
        stderr — right into the terminal the run is reporting progress in.
        """
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return
        logger.debug("viz request from %s failed", client_address, exc_info=True)

    def __init__(
        self,
        tree_getter: Callable[[], Optional[TaskTree]],
        host: str = "127.0.0.1",
        port: int = 8400,
        interval: float = 0.3,
    ):
        super().__init__((host, port), _Handler)
        self.tree_getter = tree_getter
        self.interval = interval
        self.ended = False
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self.serve_forever, name="myriapod-viz", daemon=True
        )
        self._thread.start()
        logger.info("viz server at %s", self.url)
        return self.url

    def mark_ended(self) -> None:
        """The run is over: the page flips to ENDED but stays browsable."""
        self.ended = True

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


@contextmanager
def serve(
    swarm: Any,
    port: int = 8400,
    open_browser: bool = True,
    interval: float = 0.3,
) -> Iterator[str]:
    """Context manager: live-visualize ``swarm`` runs for the block's duration.

    Yields the URL. On exit the page flips to ENDED but keeps serving
    (daemon threads) until the process exits — handy in scripts and
    notebooks.
    """
    server = VizServer(lambda: swarm.tree, port=port, interval=interval)
    url = server.start()
    if open_browser:
        webbrowser.open(url)
    try:
        yield url
    finally:
        server.mark_ended()
