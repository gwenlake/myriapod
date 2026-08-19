"""VizServer tests: real HTTP against an ephemeral port, no browser needed."""

import http.client
import json
import time

import pytest

from myriapod.core.task_tree import TaskTree
from myriapod.viz import VizServer


class SwarmStub:
    def __init__(self):
        self.tree = None


@pytest.fixture()
def server_and_stub():
    stub = SwarmStub()
    server = VizServer(lambda: stub.tree, port=0, interval=0.05)
    server.start()
    yield server, stub
    server.stop()


def _connect_events(server) -> tuple[http.client.HTTPConnection, object]:
    host, port = server.server_address[0], server.server_address[1]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    return conn, resp


def _read_event(resp, deadline: float = 5.0) -> dict:
    """Read lines until one SSE data event is complete (skips heartbeats)."""
    end = time.time() + deadline
    while time.time() < end:
        line = resp.readline().decode("utf-8").strip()
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise AssertionError("no SSE event within deadline")


def test_index_serves_the_graph_page(server_and_stub):
    server, _ = server_and_stub
    host, port = server.server_address[0], server.server_address[1]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "cytoscape" in body and "EventSource" in body
    conn.close()


def test_stream_waiting_then_snapshot_then_delta(server_and_stub):
    server, stub = server_and_stub
    conn, resp = _connect_events(server)

    assert _read_event(resp)["type"] == "waiting"

    stub.tree = TaskTree("Visualize me")
    stub.tree.decompose(
        "root",
        [{"description": "part A"}, {"description": "B", "depends_on": [1]}],
    )
    snap = _read_event(resp)
    assert snap["type"] == "snapshot"
    assert snap["goal"] == "Visualize me"
    ids = {n["id"] for n in snap["nodes"]}
    assert ids == {"root", "1", "2"}
    dep_node = next(n for n in snap["nodes"] if n["id"] == "2")
    assert dep_node["deps"] == ["1"]

    stub.tree.mark_in_progress("1", worker="w1")
    stub.tree.mark_done("1", "short digest", "RAW SECRET OUTPUT " * 50)
    delta = _read_event(resp)
    assert delta["type"] == "delta"
    changed = {n["id"]: n for n in delta["nodes"]}
    assert "1" in changed and changed["1"]["status"] == "done"
    assert changed["1"]["summary"] == "short digest"
    # Raw outputs never reach the wire.
    assert "RAW SECRET OUTPUT" not in json.dumps(delta)
    conn.close()


def test_end_event_after_mark_ended(server_and_stub):
    server, stub = server_and_stub
    stub.tree = TaskTree("run")
    stub.tree.decompose("root", [{"description": "only"}])
    conn, resp = _connect_events(server)
    assert _read_event(resp)["type"] == "snapshot"
    stub.tree.mark_done("1", "ok", "full")
    assert _read_event(resp)["type"] == "delta"
    server.mark_ended()
    assert _read_event(resp)["type"] == "end"
    conn.close()


def test_new_run_triggers_fresh_snapshot(server_and_stub):
    server, stub = server_and_stub
    stub.tree = TaskTree("first")
    conn, resp = _connect_events(server)
    first = _read_event(resp)
    assert first["type"] == "snapshot" and first["goal"] == "first"
    stub.tree = TaskTree("second")  # a new run replaces the tree
    nxt = _read_event(resp)
    assert nxt["type"] == "snapshot" and nxt["goal"] == "second"
    assert nxt["run_id"] != first["run_id"]
    conn.close()
