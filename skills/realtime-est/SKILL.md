---
name: realtime-est
description: Calibrated time estimates for work done by an AI agent (you), based on the user's real Claude Code history instead of human-scale intuition. Use BEFORE stating any duration, ETA, effort or timeline for a task you or other agents will do — "how long will this take", "ile to zajmie", "ile czasu", "when will it be done", "estimate", "wycena", sprint/milestone/roadmap planning, or a plan that lists phases with durations. Also use to report how accurate past estimates were.
---

# realtime-est — estimate agent time, not human time

Your intuition about durations comes from human-written text: "a REST API takes 2 days",
"a refactor takes a week". You execute the same work in minutes. Uncalibrated agent
estimates run **4-10x (often 100x) too long**. Never quote a duration from intuition.
Estimate the **number of agent actions** (something you can reason about), and let the
user's measured history turn actions into time.

Script: `python3 ${CLAUDE_SKILL_DIR}/scripts/rte.py` (`scripts/rte.py` in this skill's directory).
Python 3.8+ only, no dependencies. Every command imports new transcript data automatically.

## Three clocks — always keep them separate

1. **Agent work**: your own reasoning and tool calls. This is what the calibration measures.
2. **Compute / waits**: wall-clock-bound things you cannot speed up. Examples: builds, test
   suites, model training, downloads, CI, deploys, rate limits, background jobs.
3. **Human time**: the user reviewing, deciding, answering questions, testing by hand, and
   being away between sessions. For multi-day work this usually dominates.

## Procedure

1. **Look at the calibration** (once per session is enough):
   `python3 ${CLAUDE_SKILL_DIR}/scripts/rte.py profile`
   This shows the user's seconds per action, time by task size, and review time between turns.
2. **Find analogues**: `python3 ${CLAUDE_SKILL_DIR}/scripts/rte.py similar "<short task description>"`
   Past turns with their real action counts and durations are your best anchor.
3. **Write down your naive gut estimate** in seconds. Don't show it; it is logged so the
   bias can be measured.
4. **Decompose and count actions.** One action = one tool call (Read, Edit, Bash, Grep,
   WebSearch, Agent...). Count the low and high cases per step:

   | Step | Typical actions |
   |---|---|
   | Orient in an unfamiliar area (search + read) | 3-10 |
   | Change one file (read + edit) | 2-3 |
   | New file / module | 1-3 per file |
   | Run tests/build + fix loop | 2 per run, x 1-4 iterations |
   | Debugging an unknown cause | 8-30 (wide range!) |
   | Web research | 3-10 |
   | Git commit / push / PR | 2-5 |
   | Subagent fan-out | 1 per agent here; estimate the longest branch separately |

   Sanity-check the total against `similar` and the profile's size table. Widen the range
   when there is real uncertainty, such as an unknown codebase, a flaky environment, or
   unclear requirements.
5. **List compute waits** in minutes (low-high), for example "full build 3-6 min, training 2-3 h".
6. **Count human checkpoints**: how many times the user must review, approve or answer.
7. **Run the estimate**:
   `python3 ${CLAUDE_SKILL_DIR}/scripts/rte.py estimate --actions LO-HI [--compute-min A-B] [--checkpoints K]`
8. **Report** to the user: P50 and P80, with the three-clock breakdown and the action count.
   Keep it to a few lines. If they need a calendar date, add human availability on top, such
   as sessions per day or waiting for reviews from others. That part is their call; ask if
   it matters.
9. **Log it** so accuracy can be scored later. `estimate` prints a ready-made `log-estimate`
   command whose P50/P80 are already filled in; copy it as is. Fill in `--naive` (your gut
   estimate from step 3, in seconds) and `--desc`, then pick the scope:
   - `--scope next` if the user only asked a question ("how long would X take?"). The work
     starts with their next prompt.
   - `--scope current` only if you are about to do the work yourself in this same turn.

## Reporting format

```
Estimate (calibrated on your history, ~25-45 agent actions):
  agent work      ~8 min   (P80 14 min)
  build + tests   ~4 min
  your review     ~2 min   (1 checkpoint)
  total           ~15 min  (P80 ~25 min)
```

## Rules

- Never state human-scale numbers ("2-3 days", "a week") for agent work unless the profile
  supports them. If the user explicitly wants "how long would a human take", give that
  separately and label it.
- Large projects: estimate per milestone and sum them. Add roughly 1 human checkpoint per
  milestone. Very long turns (150+ actions) also go through context compaction and rework,
  so lean toward the upper half of the range.
- If `profile` says it is using the prior (little history), say the estimate is
  uncalibrated and widen the range.
- If the user asks how good the estimates are, or how long something actually took, run
  `python3 ${CLAUDE_SKILL_DIR}/scripts/rte.py accuracy`.
- Do not invent precision. Round to sensible units and always give a range.
