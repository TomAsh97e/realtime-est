"""Tests for rte.py. Run: python3 -m unittest discover -s tests"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "realtime-est" / "scripts"))

_tmp = tempfile.mkdtemp()
os.environ["RTE_HOME"] = os.path.join(_tmp, "data")
os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(_tmp, "claude")
import rte  # noqa: E402

T0 = 1_760_000_000.0


def ts(sec):
    return rte.iso(T0 + sec).replace("Z", ".000Z")


def prompt(sec, text, uuid="p1", **extra):
    e = {"type": "user", "timestamp": ts(sec), "uuid": uuid, "sessionId": "s1", "cwd": "/proj",
         "message": {"role": "user", "content": text}, "origin": {"kind": "human"}, "promptSource": "typed"}
    e.update(extra)
    return e


def tool_use(sec, name, tid, msg_id="m", tokens=10, **inp):
    return {"type": "assistant", "timestamp": ts(sec), "sessionId": "s1",
            "message": {"id": msg_id, "model": "claude-test", "usage": {"output_tokens": tokens},
                        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}}


def tool_result(sec, tid):
    return {"type": "user", "timestamp": ts(sec), "sessionId": "s1",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}}


def text(sec, msg_id="t", tokens=50):
    return {"type": "assistant", "timestamp": ts(sec), "sessionId": "s1",
            "message": {"id": msg_id, "model": "claude-test", "usage": {"output_tokens": tokens},
                        "content": [{"type": "text", "text": "done"}]}}


def notification(sec):
    return {"type": "user", "timestamp": ts(sec), "sessionId": "s1", "origin": {"kind": "task-notification"},
            "promptSource": "system", "message": {"role": "user", "content": "<task-notification>x</task-notification>"}}


class ExtractTurns(unittest.TestCase):
    def test_basic_turn(self):
        entries = [
            prompt(0, "fix the bug"),
            tool_use(2, "Read", "a", file_path="/proj/x.py"),
            tool_result(3, "a"),
            tool_use(5, "Edit", "b", msg_id="m2", file_path="/proj/x.py"),
            tool_use(5, "Edit", "c", msg_id="m2", file_path="/proj/y.py"),
            tool_result(6, "b"), tool_result(6, "c"),
            text(10),
        ]
        [t] = rte.extract_turns(entries)
        self.assertEqual(t["actions"], 3)
        self.assertEqual(t["files_edited"], 2)
        self.assertEqual(t["wall_s"], 10)
        self.assertEqual(t["active_s"], 10)
        self.assertEqual(t["out_tokens"], 10 + 10 + 50)  # m2 counted once
        self.assertEqual(t["prompt"], "fix the bug")

    def test_long_tool_and_human_wait_excluded_from_core(self):
        entries = [
            prompt(0, "train"),
            tool_use(1, "Bash", "a"), tool_result(301, "a"),           # 300 s compute
            tool_use(302, "AskUserQuestion", "q"), tool_result(362, "q"),  # 60 s human
            text(370),
        ]
        [t] = rte.extract_turns(entries)
        self.assertEqual(t["long_tool_s"], 300)
        self.assertEqual(t["human_wait_s"], 60)
        self.assertEqual(t["core_s"], 10)

    def test_notifications_split_segments(self):
        entries = [
            prompt(0, "run in background"),
            tool_use(1, "Bash", "a"), tool_result(2, "a"), text(5),
            notification(1000),  # agent idle for ~1000 s, then woken up
            tool_use(1001, "Read", "b"), tool_result(1002, "b"), text(1010),
        ]
        [t] = rte.extract_turns(entries)
        self.assertEqual(t["wall_s"], 1010)
        self.assertEqual(t["active_s"], 15)
        self.assertEqual(t["notifications"], 1)

    def test_turn_boundaries_and_noise(self):
        entries = [
            prompt(0, "<command-name>/model</command-name>", uuid="cmd"),  # local command, no agent reply
            {"type": "user", "timestamp": ts(1), "isMeta": True, "message": {"content": "meta"}},
            prompt(10, "first", uuid="u1"), text(12),
            {"type": "user", "timestamp": ts(20), "isCompactSummary": True, "message": {"content": "summary"}},
            prompt(30, "second", uuid="u2"), tool_use(31, "Bash", "a"), tool_result(33, "a"), text(40),
            {"type": "assistant", "timestamp": ts(35), "isSidechain": True, "message": {"content": []}},
        ]
        turns = rte.extract_turns(entries)
        self.assertEqual([t["prompt"] for t in turns], ["first", "second"])
        self.assertEqual(turns[1]["wall_s"], 10)

    def test_replayed_history_ignored(self):
        entries = [prompt(100, "go"), text(50), tool_use(101, "Read", "a"), tool_result(102, "a"), text(105)]
        [t] = rte.extract_turns(entries)
        self.assertEqual(t["wall_s"], 5)


class Model(unittest.TestCase):
    def test_parse_range(self):
        self.assertEqual(rte.parse_range("20-40"), (20, 40))
        self.assertEqual(rte.parse_range("40-20"), (20, 40))
        self.assertEqual(rte.parse_range("7"), (7, 7))

    def test_fit_recovers_rate(self):
        turns = [{"actions": n, "core_s": 10.0 * n} for n in [1, 2, 4, 8, 16, 32, 64, 3, 5, 9, 17, 33, 65, 6, 12]]
        m = rte.fit_model(turns)
        self.assertEqual(m["source"], "your history")
        self.assertAlmostEqual(m["b"], 1.0, places=3)
        self.assertAlmostEqual(rte.math.exp(m["a"]), 10.0, places=2)

    def test_prior_when_little_data(self):
        self.assertEqual(rte.fit_model([{"actions": 3, "core_s": 30}])["source"], "prior")

    def test_simulate_monotonic(self):
        m = rte.fit_model([])
        small = rte.simulate(m, (5, 10), None, 0, [], n=3000)
        big = rte.simulate(m, (50, 100), None, 0, [], n=3000)
        self.assertLess(small["agent"][50], big["agent"][50])
        self.assertLessEqual(small["agent"][50], small["agent"][80])

    def test_similar(self):
        turns = [{"prompt": "dodaj zmiany na githuba"}, {"prompt": "trenuj model drona"}]
        res = rte.similar("wstaw projekt na github", turns)
        self.assertEqual(res[0][1]["prompt"], "dodaj zmiany na githuba")


class Matching(unittest.TestCase):
    turns = [
        {"key": "s1:a", "session": "s1", "project": "/p", "start": rte.iso(T0), "wall_s": 100,
         "active_s": 80, "segs": [[0, 50], [70, 100]], "act": [5, 40, 75, 90], "actions": 4},
        {"key": "s1:b", "session": "s1", "project": "/p", "start": rte.iso(T0 + 200), "wall_s": 60,
         "active_s": 60, "segs": [[0, 60]], "act": [1, 2], "actions": 2},
    ]

    def test_current_scope(self):
        act = rte.match_actual({"ts": rte.iso(T0 + 30), "session": "s1", "scope": "current"}, self.turns)
        self.assertEqual(act["key"], "s1:a")
        self.assertEqual(act["active_s"], 20 + 30)
        self.assertEqual(act["actions"], 3)

    def test_next_scope(self):
        act = rte.match_actual({"ts": rte.iso(T0 + 150), "session": "s1", "scope": "next"}, self.turns)
        self.assertEqual(act["key"], "s1:b")


class Hook(unittest.TestCase):
    def test_hook_records_last_turn_and_is_silent(self):
        tr = Path(_tmp) / "t.jsonl"
        tr.write_text("\n".join(json.dumps(e) for e in [
            prompt(0, "one", uuid="u1"), text(3),
            prompt(10, "two", uuid="u2"), tool_use(11, "Read", "a"), tool_result(12, "a"), text(14),
        ]) + "\n")
        sys.stdin = io.StringIO(json.dumps({"transcript_path": str(tr), "hook_event_name": "Stop"}))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = rte.main(["hook"])
        sys.stdin = sys.__stdin__
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "")
        recs = {r["key"]: r for r in rte.load_turns()}
        self.assertIn("s1:u2", recs)
        self.assertEqual(recs["s1:u2"]["source"], "hook")

    def test_hook_survives_garbage(self):
        sys.stdin = io.StringIO("not json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = rte.main(["hook"])
        sys.stdin = sys.__stdin__
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
