# Privacy Policy: realtime-est

_Last updated: 2026-11-02_

realtime-est is an open-source Claude Code plugin. It runs entirely on your own computer.

## What data is processed

To calibrate time estimates, the plugin reads Claude Code transcripts that already exist on
your machine (`~/.claude/projects`). From them it stores the following for each turn:

- timestamps and durations (active, wall-clock, long-running commands, waiting for you)
- number and type of tool calls, and number of files edited (not file contents)
- model name and project directory path
- the first 400 characters of your prompt (used to find similar past tasks)

It also stores the estimates the agent logs: predicted durations and a short task description.

## Where it is stored

Only locally, in `~/.claude/realtime-est/`, or in the directory set by the `RTE_HOME`
environment variable.

## What is shared

Nothing. The plugin makes no network requests. It has no telemetry or analytics, and it
does not send data to the author, to Anthropic, or to any third party.

## Your control

You can delete `~/.claude/realtime-est/` at any time to remove all data. Uninstalling the
plugin stops further collection.

## Contact

Questions: https://github.com/TomAsh97e/realtime-est/issues
