#!/usr/bin/env python3
"""Summarize an arbiter event log: timeline, control time per source, stops.

    python3 safety/arbiter_report.py results/arbiter/<file>.jsonl
"""

import json
import sys
from collections import defaultdict


def main(path):
    events = [json.loads(line) for line in open(path) if line.strip()]
    if not events:
        print("no events")
        return
    t0 = events[0]["t"]
    print(f"{'t [s]':>8}  event")
    print("-" * 60)
    for e in events:
        extra = {k: v for k, v in e.items() if k not in ("t", "event")}
        print(f"{e['t'] - t0:>8.2f}  {e['event']:<10} {extra if extra else ''}")

    decisions = [e for e in events if e["event"] == "decision"]
    per_source = defaultdict(float)
    for a, b in zip(decisions, decisions[1:] + [{"t": events[-1]["t"]}]):
        per_source[a["source"]] += b["t"] - a["t"]
    stops = defaultdict(int)
    for d in decisions:
        if d["source"] == "stop":
            stops[d["reason"]] += 1
    standstills = [e["after_s"] for e in events if e["event"] == "standstill"]

    print("\ncontrol time by source:")
    total = sum(per_source.values()) or 1.0
    for s, v in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<9} {v:7.1f} s  ({100 * v / total:4.1f}%)")
    print("\nstops by reason:")
    for r, n in sorted(stops.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {r}")
    if standstills:
        print(f"\ntime to standstill: max {max(standstills):.2f} s, "
              f"mean {sum(standstills) / len(standstills):.2f} s over {len(standstills)} stops")
    takeovers = sum(1 for e in events if e["event"] == "takeover")
    print(f"operator takeovers: {takeovers}")


if __name__ == "__main__":
    main(sys.argv[1])
