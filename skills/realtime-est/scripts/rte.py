#!/usr/bin/env python3
"""realtime-est: calibrate AI-agent time estimates against your own Claude Code history.

LLMs learned "how long things take" from human-written text, so they quote human
timelines (hours, days) for work an agent finishes in minutes. This tool measures
how long *your* agent actually takes, per agent action, and turns an estimate of
the number of actions into a calibrated time range.

Standard library only. Data stays local in ~/.claude/realtime-est/.

Subcommands:
  hook          Stop-hook entry point (reads hook JSON on stdin, records the turn)
  backfill      Import past turns from Claude Code transcripts
  profile       Show your calibration profile
  similar TEXT  Past turns most similar to a task description
  estimate      Calibrated time range for N agent actions (+ compute, + human checkpoints)
  log-estimate  Record an estimate so it can be scored later
  done          Close a multi-turn task estimate and score it
  accuracy      Compare logged estimates with what actually happened
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
DATA_DIR = Path(os.environ.get("RTE_HOME") or CLAUDE_DIR / "realtime-est")
TURNS_FILE = DATA_DIR / "turns.jsonl"
ESTIMATES_FILE = DATA_DIR / "estimates.jsonl"
INDEX_FILE = DATA_DIR / "backfill_index.json"

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SUBAGENT_TOOLS = {"Agent", "Task"}
HUMAN_TOOLS = {"AskUserQuestion"}  # tool time spent waiting for the human
LONG_TOOL_S = 60.0  # a single tool call longer than this counts as compute, not agent work
MAX_REVIEW_GAP_S = 30 * 60  # gaps longer than this between turns are breaks, not reviews

# Prior used until there are enough of your own turns (log-log fit: core_s = e^a * actions^b).
PRIOR = {"a": math.log(9.0), "b": 0.95, "resid": [-0.9, -0.5, -0.2, 0.0, 0.2, 0.45, 0.8, 1.2]}
MIN_FIT_TURNS = 15


# --------------------------------------------------------------------------- utils

def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fmt_dur(s):
    if s is None:
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}min" if s < 600 else f"{s / 60:.0f}min"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def parse_range(s, cast=float):
    """'20-40' -> (20, 40); '30' -> (30, 30)."""
    if s is None:
        return None
    s = str(s).strip()
    m = re.fullmatch(r"([\d.]+)\s*(?:-|\.\.|to)\s*([\d.]+)", s)
    if m:
        lo, hi = cast(m.group(1)), cast(m.group(2))
        return (min(lo, hi), max(lo, hi))
    v = cast(s)
    return (v, v)


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return out


# ---------------------------------------------------------- transcript -> turns

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content):
            return None
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return None


def classify_user(e):
    """Return 'prompt' (a human started a turn), 'notification' (background work woke the
    agent up inside the same task) or None (tool results, meta, summaries, noise)."""
    if e.get("type") != "user" or e.get("isSidechain") or e.get("isMeta") or e.get("isCompactSummary"):
        return None
    text = _text_of((e.get("message") or {}).get("content"))
    if text is None:
        return None
    t = text.lstrip()
    if not t:
        return None
    origin = e.get("origin") if isinstance(e.get("origin"), dict) else {}
    kind = origin.get("kind")
    if t.startswith("<task-notification>"):
        return "notification"
    if kind and kind != "human":
        return "notification"
    if kind == "human":
        return "prompt"
    if t.startswith(("<local-command", "[Request interrupted", "Caveat:", "<system-reminder>")):
        return None
    if e.get("promptSource") == "system":
        return "notification"
    return "prompt"


class _Turn:
    def __init__(self, e, t):
        text = _text_of(e["message"].get("content")) or ""
        self.key = f"{e.get('sessionId', '?')}:{e.get('uuid', '?')}"
        self.session = e.get("sessionId")
        self.project = e.get("cwd")
        self.start = t
        self.prompt = re.sub(r"\s+", " ", text).strip()[:400]
        self.last = t
        self.seg_start = t
        self.seg_end = None
        self.segs = []
        self.tools = Counter()
        self.files = set()
        self.action_offsets = []
        self.pending = {}  # tool_use_id -> (start_ts, name)
        self.long_tool_s = 0.0
        self.human_wait_s = 0.0
        self.notifications = 0
        self.compactions = 0
        self.tokens_by_msg = {}
        self.assistant_entries = 0
        self.models = Counter()
        self.cc_turn_ms = 0

    def touch(self, t):
        if t > self.last:
            self.last = t
        if self.seg_end is None or t > self.seg_end:
            self.seg_end = t

    def close_seg(self):
        if self.seg_end is not None and self.seg_end > self.seg_start:
            self.segs.append((self.seg_start, self.seg_end))
        self.seg_end = None

    def finish(self):
        self.close_seg()
        active = sum(e - s for s, e in self.segs)
        core = max(0.0, active - self.long_tool_s - self.human_wait_s)
        actions = sum(self.tools.values())
        return {
            "key": self.key,
            "session": self.session,
            "project": self.project,
            "start": iso(self.start),
            "wall_s": round(self.last - self.start, 1),
            "active_s": round(active, 1),
            "core_s": round(core, 1),
            "long_tool_s": round(self.long_tool_s, 1),
            "human_wait_s": round(self.human_wait_s, 1),
            "actions": actions,
            "tools": dict(self.tools),
            "files_edited": len(self.files),
            "subagents": sum(self.tools.get(n, 0) for n in SUBAGENT_TOOLS),
            "notifications": self.notifications,
            "compactions": self.compactions,
            "out_tokens": sum(self.tokens_by_msg.values()),
            "model": self.models.most_common(1)[0][0] if self.models else None,
            "cc_turn_s": round(self.cc_turn_ms / 1000, 1) if self.cc_turn_ms else None,
            "segs": [[round(s - self.start), round(e - self.start)] for s, e in self.segs],
            "act": [round(o) for o in self.action_offsets],
            "prompt": self.prompt,
        }


def extract_turns(entries):
    """Split transcript entries (file order) into human-initiated turns."""
    turns = []
    cur = None
    for e in entries:
        if not isinstance(e, dict) or e.get("isSidechain"):
            continue
        t = parse_ts(e.get("timestamp"))
        kind = classify_user(e)
        if kind == "prompt" and t is not None:
            if cur and cur.assistant_entries:
                turns.append(cur.finish())
            cur = _Turn(e, t)
            continue
        if cur is None or t is None or t < cur.start - 1:
            continue  # before the first prompt, or replayed pre-compaction history
        typ = e.get("type")
        if kind == "notification":
            cur.close_seg()
            cur.seg_start = t
            cur.notifications += 1
            cur.last = max(cur.last, t)
            continue
        if typ == "assistant":
            msg = e.get("message") or {}
            cur.touch(t)
            cur.assistant_entries += 1
            if msg.get("model") and not str(msg["model"]).startswith("<"):
                cur.models[msg["model"]] += 1
            usage = msg.get("usage") or {}
            mid = msg.get("id") or id(msg)
            cur.tokens_by_msg[mid] = max(cur.tokens_by_msg.get(mid, 0), int(usage.get("output_tokens") or 0))
            for c in msg.get("content") or []:
                if not isinstance(c, dict) or c.get("type") != "tool_use":
                    continue
                name = c.get("name") or "?"
                cur.tools[name] += 1
                cur.action_offsets.append(t - cur.start)
                if c.get("id"):
                    cur.pending[c["id"]] = (t, name)
                if name in EDIT_TOOLS:
                    inp = c.get("input") or {}
                    p = inp.get("file_path") or inp.get("notebook_path")
                    if p:
                        cur.files.add(p)
        elif typ == "user":
            content = (e.get("message") or {}).get("content")
            if isinstance(content, list):
                cur.touch(t)
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        started = cur.pending.pop(c.get("tool_use_id"), None)
                        if started:
                            dur = t - started[0]
                            if started[1] in HUMAN_TOOLS:
                                cur.human_wait_s += dur
                            elif dur > LONG_TOOL_S:
                                cur.long_tool_s += dur
        elif typ == "system":
            st = e.get("subtype")
            if st == "turn_duration":
                cur.cc_turn_ms += int(e.get("durationMs") or 0)
            elif st == "compact_boundary":
                cur.compactions += 1
    if cur and cur.assistant_entries:  # local commands like /model never reach the agent
        turns.append(cur.finish())
    return turns


def read_transcript(path):
    return read_jsonl(path)


def read_transcript_tail(path, max_bytes=256 * 1024 * 1024):
    """Read the end of a transcript, growing the window until it contains a human prompt."""
    size = os.path.getsize(path)
    window = 2 * 1024 * 1024
    while True:
        start = max(0, size - window)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
        lines = data.split(b"\n")
        if start > 0:
            lines = lines[1:]  # first line is probably partial
        entries = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                entries.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        if start == 0 or window >= max_bytes or any(classify_user(e) == "prompt" for e in entries):
            return entries
        window *= 4


# ------------------------------------------------------------------- storage

def load_turns(include_chat=True):
    by_key = {}
    for r in read_jsonl(TURNS_FILE):
        if "key" in r:
            by_key[r["key"]] = r
    turns = sorted(by_key.values(), key=lambda r: r.get("start", ""))
    if not include_chat:
        turns = [r for r in turns if r.get("actions", 0) > 0]
    return turns


def _same(a, b):
    return a.get("wall_s") == b.get("wall_s") and a.get("actions") == b.get("actions")


def store_turns(new_turns, source, existing=None):
    if existing is None:
        existing = {r["key"]: r for r in load_turns()}
    n = 0
    for r in new_turns:
        old = existing.get(r["key"])
        if old and _same(old, r):
            continue
        r["source"] = source
        append_jsonl(TURNS_FILE, r)
        existing[r["key"]] = r
        n += 1
    return n


def transcript_files():
    root = CLAUDE_DIR / "projects"
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*/*.jsonl") if p.is_file())


def backfill(quiet=False):
    try:
        index = json.loads(INDEX_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        index = {}
    existing = {r["key"]: r for r in load_turns()}
    added = scanned = 0
    for p in transcript_files():
        st = p.stat()
        sig = [int(st.st_mtime), st.st_size]
        if index.get(str(p)) == sig:
            continue
        scanned += 1
        added += store_turns(extract_turns(read_transcript(p)), "backfill", existing)
        index[str(p)] = sig
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index))
    if not quiet:
        print(f"backfill: scanned {scanned} changed transcript(s), stored {added} turn record(s); "
              f"{len(existing)} turns total in {TURNS_FILE}")
    return added


# ------------------------------------------------------------------- model

def fit_model(turns):
    pts = [(math.log(r["actions"]), math.log(max(r["core_s"], 1.0)))
           for r in turns if r.get("actions", 0) >= 1 and r.get("core_s", 0) > 0]
    if len(pts) < MIN_FIT_TURNS:
        return dict(PRIOR, n=len(pts), source="prior")
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return dict(PRIOR, n=len(pts), source="prior")
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    b = min(max(b, 0.5), 1.3)  # keep the slope physically sensible
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in pts]
    return {"a": a, "b": b, "resid": resid, "n": len(pts), "source": "your history"}


def review_gaps(turns):
    """Seconds the human spends between the end of one turn and their next prompt."""
    by_session = {}
    for r in turns:
        by_session.setdefault(r.get("session"), []).append(r)
    gaps = []
    for rs in by_session.values():
        rs.sort(key=lambda r: r["start"])
        for prev, nxt in zip(rs, rs[1:]):
            end = parse_ts(prev["start"]) + prev["wall_s"]
            g = parse_ts(nxt["start"]) - end
            if 3 <= g <= MAX_REVIEW_GAP_S:
                gaps.append(g)
    return gaps


def load_estimates():
    """Estimates in log order, with `done_ts` attached to closed multi-turn tasks."""
    ests, done = [], {}
    for r in read_jsonl(ESTIMATES_FILE):
        if r.get("done_for"):
            done[r["done_for"]] = r["ts"]
        else:
            ests.append(r)
    for e in ests:
        if e.get("id") in done:
            e["done_ts"] = done[e["id"]]
    return ests


def estimates_with_actuals(turns):
    return [(est, match_actual(est, turns)) for est in load_estimates()]


def action_bias(turns):
    """Median of actual/estimated actions over scored estimates (>=5), else 1.0."""
    ratios = []
    for est, act in estimates_with_actuals(turns):
        if act and est.get("actions_mid") and act["actions"] > 0:
            ratios.append(act["actions"] / est["actions_mid"])
    if len(ratios) < 5:
        return 1.0, len(ratios)
    return min(max(statistics.median(ratios), 0.33), 3.0), len(ratios)


def simulate(model, actions, compute, checkpoints, gaps, bias=1.0, n=20000, seed=7):
    rng = random.Random(seed)
    lo, hi = actions
    lo, hi = max(lo, 1) * bias, max(hi, 1) * bias
    # treat lo/hi as P10/P90 of a lognormal on the number of actions
    mu = (math.log(lo) + math.log(hi)) / 2
    sd = (math.log(hi) - math.log(lo)) / (2 * 1.2816) if hi > lo else 0.0
    resid = model["resid"]
    agent, comp, human, work, total = [], [], [], [], []
    for _ in range(n):
        k = max(1.0, math.exp(rng.gauss(mu, sd)))
        a = math.exp(model["a"] + model["b"] * math.log(k) + rng.choice(resid))
        c = rng.uniform(*compute) if compute else 0.0
        h = sum(rng.choice(gaps) for _ in range(checkpoints)) if (checkpoints and gaps) else checkpoints * 120.0
        agent.append(a)
        comp.append(c)
        human.append(h)
        work.append(a + c)
        total.append(a + c + h)
    q = lambda xs: {p: quantile(xs, p / 100) for p in (10, 50, 80, 90)}
    return {"agent": q(agent), "compute": q(comp), "human": q(human), "work": q(work), "total": q(total)}


# ------------------------------------------------------------------- matching

def match_actual(est, turns):
    t = parse_ts(est.get("ts"))
    if t is None:
        return None
    sess = est.get("session")
    if est.get("scope") == "task":
        t1 = parse_ts(est.get("done_ts"))
        if t1 is None:
            return None  # still open
        cands = [r for r in turns if _same_place(est, r)]
        return window_actual(cands, t, t1)
    cands = [r for r in turns if (sess and r.get("session") == sess) or (not sess and r.get("project") == est.get("cwd"))]
    if est.get("scope") == "current":
        for r in cands:
            s = parse_ts(r["start"])
            if s - 1 <= t <= s + r["wall_s"] + 5:
                off = t - s
                active = sum(max(0, e - max(a, off)) for a, e in r.get("segs", []))
                acts = sum(1 for o in r.get("act", []) if o >= off)
                return {"key": r["key"], "active_s": active, "actions": acts, "wall_s": r["wall_s"] - off}
        return None
    nxt = [r for r in cands if parse_ts(r["start"]) > t + 1]
    if not nxt:
        return None
    r = min(nxt, key=lambda r: r["start"])
    return {"key": r["key"], "active_s": r["active_s"], "actions": r["actions"], "wall_s": r["wall_s"]}


def _same_place(est, r):
    """A multi-turn task may span sessions: match by session or by project directory."""
    if est.get("session") and r.get("session") == est.get("session"):
        return True
    proj, cwd = r.get("project"), est.get("cwd")
    return bool(proj and cwd and (cwd == proj or cwd.startswith(proj.rstrip(os.sep) + os.sep)))


def window_actual(turns, t0, t1):
    """Agent-active time and actions of all turns inside the window [t0, t1]."""
    active, acts, keys = 0.0, 0, []
    for r in turns:
        s = parse_ts(r["start"])
        if s > t1 or s + r["wall_s"] < t0:
            continue
        part = sum(max(0.0, min(s + e, t1) - max(s + a, t0)) for a, e in r.get("segs", []))
        n = sum(1 for o in r.get("act", []) if t0 <= s + o <= t1)
        if part > 0 or n:
            active += part
            acts += n
            keys.append(r["key"])
    return {"key": ",".join(keys), "turns": len(keys), "active_s": active, "actions": acts, "wall_s": t1 - t0}


# ------------------------------------------------------------------- similarity

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _tokens(text):
    return {w.lower()[:5] for w in _WORD.findall(text or "")}  # crude stem, works for inflected languages too


def similar(text, turns, k=8):
    q = _tokens(text)
    if not q:
        return []
    docs = [(r, _tokens(r.get("prompt"))) for r in turns]
    df = Counter(t for _, toks in docs for t in toks)
    n = len(docs) or 1
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}
    scored = []
    for r, toks in docs:
        inter = q & toks
        if not inter:
            continue
        num = sum(idf.get(t, 1) for t in inter)
        den = math.sqrt(sum(idf.get(t, 1) for t in q) * sum(idf.get(t, 1) for t in toks)) or 1
        scored.append((num / den, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# ------------------------------------------------------------------- commands

def cmd_hook(_args):
    """Stop hook. Must never fail or print to stdout."""
    try:
        payload = json.load(sys.stdin)
        path = payload.get("transcript_path")
        if not path or not os.path.exists(path):
            return 0
        turns = extract_turns(read_transcript_tail(path))
        if turns:
            store_turns([turns[-1]], "hook")
    except Exception as exc:  # noqa: BLE001 - a hook must never break the session
        try:
            append_jsonl(DATA_DIR / "errors.log", {"ts": iso(time.time()), "error": repr(exc)})
        except Exception:  # noqa: BLE001
            pass
    return 0


def cmd_backfill(_args):
    backfill()
    return 0


def _ensure_data(args):
    if not getattr(args, "no_backfill", False):
        backfill(quiet=True)


def cmd_profile(args):
    _ensure_data(args)
    turns = load_turns()
    work = [r for r in turns if r.get("actions", 0) > 0]
    chat = [r for r in turns if r.get("actions", 0) == 0]
    if args.project:
        work = [r for r in work if r.get("project") == os.path.abspath(args.project)]
    model = fit_model(work)
    gaps = review_gaps(turns)
    bias, nb = action_bias(turns)
    if args.json:
        print(json.dumps({
            "turns": len(turns), "work_turns": len(work),
            "model": {k: model[k] for k in ("a", "b", "n", "source")},
            "sec_per_action_p50": _spa(work, 0.5),
            "review_gap_p50_s": quantile(gaps, 0.5),
            "action_bias": bias, "scored_estimates": nb,
        }, indent=2))
        return 0
    if not turns:
        print("No history yet. Use Claude Code for a while (or check ~/.claude/projects) and run `backfill`.")
        print("Until then estimates use a built-in prior.")
        return 0
    first, last = turns[0]["start"][:10], turns[-1]["start"][:10]
    models = Counter(r.get("model") for r in work if r.get("model"))
    print(f"History: {len(turns)} turns ({len(work)} with tool use, {len(chat)} chat-only), {first} .. {last}")
    if models:
        print("Models:  " + ", ".join(f"{m} ({c})" for m, c in models.most_common(4)))
    print()
    print("Agent time by size (core = active time minus long-running commands and waiting for you)")
    print(f"{'actions':>10} {'turns':>6} {'core P50':>9} {'core P80':>9} {'wall P50':>9} {'wall P80':>9} {'files P50':>9}")
    for lo, hi in [(1, 5), (6, 15), (16, 40), (41, 100), (101, 250), (251, 10 ** 9)]:
        b = [r for r in work if lo <= r["actions"] <= hi]
        if not b:
            continue
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        cs, ws = [r["core_s"] for r in b], [r["wall_s"] for r in b]
        fe = quantile([r.get("files_edited", 0) for r in b], 0.5)
        print(f"{label:>10} {len(b):>6} {fmt_dur(quantile(cs, .5)):>9} {fmt_dur(quantile(cs, .8)):>9} "
              f"{fmt_dur(quantile(ws, .5)):>9} {fmt_dur(quantile(ws, .8)):>9} {fe:>9.0f}")
    if chat:
        print(f"{'0 (chat)':>10} {len(chat):>6} {fmt_dur(quantile([r['wall_s'] for r in chat], .5)):>9}")
    print()
    spa = _spa(work, 0.5)
    print(f"Seconds per agent action: P50 {spa:.1f}s, P80 {_spa(work, 0.8):.1f}s" if spa else "")
    print(f"Fit: core_s = {math.exp(model['a']):.1f} * actions^{model['b']:.2f}  (n={model['n']}, {model['source']})")
    if gaps:
        print(f"Your review/response time between turns: P50 {fmt_dur(quantile(gaps, .5))}, "
              f"P80 {fmt_dur(quantile(gaps, .8))} (n={len(gaps)})")
    lt = [r["long_tool_s"] for r in work if r.get("long_tool_s", 0) > 0]
    if lt:
        print(f"Turns with long-running commands (>{LONG_TOOL_S:.0f}s each): {len(lt)}, "
              f"P50 {fmt_dur(quantile(lt, .5))} of compute per such turn")
    if nb:
        print(f"Scored estimates: {nb}; action-count bias (actual/estimated) applied: x{bias:.2f}")
    return 0


def _spa(work, q):
    xs = [r["core_s"] / r["actions"] for r in work if r.get("actions", 0) >= 3 and r.get("core_s", 0) > 0]
    return quantile(xs, q)


def cmd_similar(args):
    _ensure_data(args)
    turns = [r for r in load_turns() if r.get("actions", 0) > 0 or args.include_chat]
    res = similar(" ".join(args.text), turns, k=args.k)
    if args.json:
        print(json.dumps([{"score": round(s, 3), **{k: r.get(k) for k in (
            "start", "project", "actions", "core_s", "wall_s", "files_edited", "prompt")}} for s, r in res], indent=2,
            ensure_ascii=False))
        return 0
    if not res:
        print("No similar past turns found.")
        return 0
    print(f"{'score':>5} {'actions':>7} {'core':>7} {'wall':>7} {'files':>5}  prompt")
    for s, r in res:
        p = r.get("prompt", "")
        print(f"{s:5.2f} {r['actions']:>7} {fmt_dur(r['core_s']):>7} {fmt_dur(r['wall_s']):>7} "
              f"{r.get('files_edited', 0):>5}  {p[:90]}{'…' if len(p) > 90 else ''}")
    return 0


def cmd_estimate(args):
    _ensure_data(args)
    turns = load_turns()
    work = [r for r in turns if r.get("actions", 0) > 0]
    model = fit_model(work)
    gaps = review_gaps(turns)
    actions = parse_range(args.actions)
    compute = parse_range(args.compute_min) if args.compute_min else None
    if compute:
        compute = (compute[0] * 60, compute[1] * 60)
    bias, nb = action_bias(turns) if not args.no_bias else (1.0, 0)
    res = simulate(model, actions, compute, args.checkpoints, gaps, bias)
    if args.json:
        print(json.dumps({"actions": actions, "bias": bias, "model_source": model["source"],
                          "model_n": model["n"], **res}, indent=2))
        return 0
    print(f"Calibrated estimate for {actions[0]:g}-{actions[1]:g} agent actions"
          + (f" (x{bias:.2f} action bias from {nb} scored estimates)" if bias != 1.0 else ""))
    print(f"Model: {model['source']} (n={model['n']})")
    print(f"{'':>16} {'P10':>8} {'P50':>8} {'P80':>8} {'P90':>8}")
    rows = [("agent work", "agent")]
    if compute:
        rows.append(("compute/waits", "compute"))
    if args.checkpoints:
        rows.append((f"you ({args.checkpoints} chkpt)", "human"))
    rows.append(("TOTAL wall-clock", "total"))
    for label, k in rows:
        q = res[k]
        print(f"{label:>16} " + " ".join(f"{fmt_dur(q[p]):>8}" for p in (10, 50, 80, 90)))
    if model["source"] == "prior":
        print("\nNote: fewer than %d of your turns recorded - using a generic prior. Run `backfill`." % MIN_FIT_TURNS)
    w = res["work"]
    print("\nLog it (scored against agent work + compute, i.e. without your review time):")
    print(f"  python3 {Path(__file__).resolve()} log-estimate --p50 {w[50]:.0f} --p80 {w[80]:.0f} "
          f"--actions {actions[0]:g}-{actions[1]:g} --naive <gut-feel seconds> --scope <current|next|task> --desc \"<task>\"")
    return 0


def cmd_log_estimate(args):
    actions = parse_range(args.actions) if args.actions else None
    rec = {
        "id": f"{int(time.time()) % 10 ** 6:06d}{random.randrange(100):02d}",
        "ts": iso(time.time()),
        "session": os.environ.get("CLAUDE_CODE_SESSION_ID"),
        "cwd": os.getcwd(),
        "scope": args.scope,
        "desc": args.desc,
        "actions": actions,
        "actions_mid": math.sqrt(actions[0] * actions[1]) if actions else None,
        "p50_s": args.p50,
        "p80_s": args.p80,
        "naive_s": args.naive,
    }
    append_jsonl(ESTIMATES_FILE, rec)
    what = {"current": "rest of this turn", "next": "next turn", "task": "multi-turn task"}[args.scope]
    print(f"logged estimate {rec['id']} ({what}): P50 {fmt_dur(args.p50)}, P80 {fmt_dur(args.p80)}")
    if args.scope == "task":
        print(f"When the task is finished run: python3 {Path(__file__).resolve()} done")
    return 0


def cmd_done(args):
    """Close a multi-turn task estimate (default: the latest open one for this session/project)."""
    _ensure_data(args)
    here = {"session": os.environ.get("CLAUDE_CODE_SESSION_ID"), "cwd": os.getcwd()}
    open_tasks = [e for e in load_estimates() if e.get("scope") == "task" and not e.get("done_ts")]
    if args.id:
        open_tasks = [e for e in open_tasks if e.get("id") == args.id]
    else:
        open_tasks = [e for e in open_tasks
                      if _same_place(here, {"session": e.get("session"), "project": e.get("cwd")})
                      or _same_place(e, {"session": here["session"], "project": here["cwd"]})] or open_tasks
    if not open_tasks:
        print("No open task estimate found.")
        return 1
    est = open_tasks[-1]
    now = iso(time.time())
    append_jsonl(ESTIMATES_FILE, {"done_for": est["id"], "ts": now})
    est["done_ts"] = now
    act = match_actual(est, load_turns())
    print(f"closed {est['id']}: {est.get('desc') or ''}")
    if act:
        ratio = act["active_s"] / est["p50_s"] if est.get("p50_s") else None
        print(f"  estimate P50 {fmt_dur(est.get('p50_s'))} / P80 {fmt_dur(est.get('p80_s'))}; actual agent+compute "
              f"{fmt_dur(act['active_s'])} over {act['turns']} turn(s), {act['actions']} actions, "
              f"{fmt_dur(act['wall_s'])} wall-clock" + (f"; actual/P50 = {ratio:.2f}" if ratio else ""))
    return 0


def cmd_accuracy(args):
    _ensure_data(args)
    turns = load_turns()
    pairs = estimates_with_actuals(turns)
    if not pairs:
        print("No logged estimates yet.")
        return 0
    print(f"{'when':>16} {'scope':>7} {'P50':>7} {'P80':>7} {'actual':>7} {'ratio':>6} {'<=P80':>5} "
          f"{'est.act':>8} {'act':>5} {'naive':>7}  desc")
    ratios, cover, naive_ratios = [], [], []
    for est, act in pairs[-args.last:]:
        a = act["active_s"] if act else None
        if act is None and est.get("scope") == "task":
            print(f"{est['ts'][:16]:>16} {'task':>7} {fmt_dur(est.get('p50_s')):>7} {fmt_dur(est.get('p80_s')):>7} "
                  f"{'open':>7}  (close with `done`)  {(est.get('desc') or '')[:50]}")
            continue
        ratio = a / est["p50_s"] if (a is not None and est.get("p50_s")) else None
        if ratio is not None:
            ratios.append(ratio)
            cover.append(a <= est["p80_s"] if est.get("p80_s") else False)
            if est.get("naive_s"):
                naive_ratios.append(a / est["naive_s"])
        ea = est.get("actions")
        print(f"{est['ts'][:16]:>16} {est['scope']:>7} {fmt_dur(est.get('p50_s')):>7} {fmt_dur(est.get('p80_s')):>7} "
              f"{fmt_dur(a):>7} {(f'{ratio:.2f}' if ratio else '-'):>6} "
              f"{('yes' if ratio and a <= (est.get('p80_s') or 0) else ('-' if ratio is None else 'no')):>5} "
              f"{(f'{ea[0]:g}-{ea[1]:g}' if ea else '-'):>8} {(act['actions'] if act else '-'):>5} "
              f"{fmt_dur(est.get('naive_s')):>7}  {(est.get('desc') or '')[:50]}")
    if ratios:
        print(f"\nactual/P50: median {statistics.median(ratios):.2f} (1.0 = perfect; >1 = underestimate); "
              f"P80 coverage {sum(cover)}/{len(cover)} (target ~80%)")
    if naive_ratios:
        print(f"actual/naive (the uncalibrated first guess): median {statistics.median(naive_ratios):.3f} "
              f"-> naive guesses were ~{1 / statistics.median(naive_ratios):.0f}x too long")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="rte", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hook", help="Stop-hook entry point (stdin: hook JSON)").set_defaults(fn=cmd_hook)
    sub.add_parser("backfill", help="import past turns from transcripts").set_defaults(fn=cmd_backfill)

    sp = sub.add_parser("profile", help="show your calibration profile")
    sp.add_argument("--project", help="only turns from this project directory")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-backfill", action="store_true")
    sp.set_defaults(fn=cmd_profile)

    sp = sub.add_parser("similar", help="past turns similar to a task description")
    sp.add_argument("text", nargs="+")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--include-chat", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-backfill", action="store_true")
    sp.set_defaults(fn=cmd_similar)

    sp = sub.add_parser("estimate", help="calibrated time range for N agent actions")
    sp.add_argument("--actions", required=True, help="expected agent actions (tool calls), e.g. 20-40")
    sp.add_argument("--compute-min", help="wall-clock-bound work in minutes (builds, training, CI), e.g. 5-15")
    sp.add_argument("--checkpoints", type=int, default=0, help="times the human must review/answer")
    sp.add_argument("--no-bias", action="store_true", help="ignore learned action-count bias")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-backfill", action="store_true")
    sp.set_defaults(fn=cmd_estimate)

    sp = sub.add_parser("log-estimate", help="record an estimate for later scoring")
    sp.add_argument("--p50", type=float, required=True, help="seconds")
    sp.add_argument("--p80", type=float, required=True, help="seconds")
    sp.add_argument("--actions", help="estimated actions range, e.g. 20-40")
    sp.add_argument("--naive", type=float, help="the uncalibrated (human-scale) guess, seconds")
    sp.add_argument("--scope", choices=["current", "next", "task"], default="current",
                    help="current: the work continues in this turn; next: user will start it in the next prompt; "
                         "task: spans several turns/sessions until closed with `done`")
    sp.add_argument("--desc", default="")
    sp.set_defaults(fn=cmd_log_estimate)

    sp = sub.add_parser("done", help="close a multi-turn task estimate and score it")
    sp.add_argument("--id", help="estimate id (default: latest open task here)")
    sp.add_argument("--no-backfill", action="store_true")
    sp.set_defaults(fn=cmd_done)

    sp = sub.add_parser("accuracy", help="score logged estimates against reality")
    sp.add_argument("--last", type=int, default=20)
    sp.add_argument("--no-backfill", action="store_true")
    sp.set_defaults(fn=cmd_accuracy)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
