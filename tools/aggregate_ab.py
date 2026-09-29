#!/usr/bin/env python3
"""Aggregate baseline vs terrain-layer runs into one A/B comparison.

A run is included only if it has a scripted route result. Localization
metrics are joined by label when that run's samples.csv exists.

    source /opt/ros/jazzy/setup.bash
    python3 tools/aggregate_ab.py                       # writes docs/results/ab/
    python3 tools/aggregate_ab.py --exclude terrain_run3

Outputs: ab_summary.md (tables for the README), ab_summary.json, ab_figure.png
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import localization_eval as le  # noqa: E402

STAMP = re.compile(r"^\d{8}_\d{6}_")
LOST_CM = 200.0  # AMCL error above this counts as a localization loss
DEFAULT_EXCLUDE = ["terrain_run3"]  # default initial pose bug teleported AMCL


def label_of(name: str) -> str:
    return STAMP.sub("", name)


def condition_of(label: str) -> str:
    return label.split("_run")[0]


def load_routes(route_dir: pathlib.Path, exclude: set) -> dict:
    runs = {}
    for f in sorted(route_dir.glob("*.json")):
        label = label_of(f.stem)
        if label in exclude:
            continue
        runs[label] = json.loads(f.read_text())  # later files win on duplicate labels
    return runs


def load_localization(loc_dir: pathlib.Path, labels: set) -> dict:
    out = {}
    for d in sorted(loc_dir.iterdir()):
        csv_path = d / "samples.csv"
        if not d.is_dir() or not csv_path.exists():
            continue
        label = label_of(d.name)
        if label not in labels:
            continue
        rows = le._load_csv(str(csv_path))
        if rows:
            out[label] = rows
    return out


def localization_metrics(rows: list) -> dict:
    err = le._f(rows, "amcl_pos_err") * 100
    regions = np.array([r["region"] for r in rows])
    m = {
        "amcl_median_cm": float(np.median(err)),
        "amcl_p95_cm": float(np.percentile(err, 95)),
        "amcl_max_cm": float(err.max()),
        "lost": bool(err.max() > LOST_CM),
    }
    rough = regions == "rough_iso_f"
    if rough.any():
        m["amcl_rough_median_cm"] = float(np.median(err[rough]))
        t = le._f(rows, "t")
        dt_ = np.diff(t, prepend=t[0])
        m["rough_exposure_s"] = float(dt_[rough].sum())
    if le._has_tilt(rows):
        tilt = le._f(rows, "tilt_deg")
        m["tilt_p95_deg"] = float(np.percentile(tilt, 95))
        m["tilt_max_deg"] = float(tilt.max())
    return m


def summarize_condition(labels: list, routes: dict, loc: dict) -> dict:
    legs = [l["name"] for l in routes[labels[0]]["legs"]]
    leg_ok = {name: [] for name in legs}
    leg_time = {name: [] for name in legs}
    full, recov, false_ok, route_time = [], [], [], []
    for lab in labels:
        run_legs = {l["name"]: l for l in routes[lab]["legs"]}
        ok_all = True
        for name in legs:
            l = run_legs.get(name)
            ok = l is not None and l["result"] == "succeeded"
            leg_ok[name].append(ok)
            ok_all &= ok
            if ok:
                leg_time[name].append(l["sim_seconds"])
        full.append(ok_all)
        recov.append(sum((l.get("recoveries") or 0) for l in routes[lab]["legs"]))
        false_ok.append(sum(1 for l in routes[lab]["legs"] if l["result"] == "false_success"))
        route_time.append(sum(l["sim_seconds"] for l in routes[lab]["legs"]))

    locm = {lab: localization_metrics(loc[lab]) for lab in labels if lab in loc}
    episodes, controls = [], []
    for lab in labels:
        if lab in loc:
            epi = le.episode_analysis(loc[lab])
            if epi:
                episodes += [e["delta_cm"] for e in epi["episodes"]]
                controls += epi["controls"]
    tilt_test = le.compare(episodes, controls) if episodes and controls else None

    def med(key):
        v = [m[key] for m in locm.values() if key in m]
        return float(np.median(v)) if v else None

    return {
        "runs": labels,
        "n_runs": len(labels),
        "full_route_success": int(sum(full)),
        "leg_success": {k: [int(sum(v)), len(v)] for k, v in leg_ok.items()},
        "leg_median_time_s": {k: (float(np.median(v)) if v else None) for k, v in leg_time.items()},
        "recoveries_median": float(np.median(recov)),
        "false_successes_total": int(sum(false_ok)),
        "route_sim_seconds_median": float(np.median(route_time)),
        "localized_runs": len(locm),
        "amcl_median_cm": med("amcl_median_cm"),
        "amcl_p95_cm": med("amcl_p95_cm"),
        "amcl_rough_median_cm": med("amcl_rough_median_cm"),
        "rough_exposure_s": med("rough_exposure_s"),
        "tilt_p95_deg": med("tilt_p95_deg"),
        "runs_lost": int(sum(m["lost"] for m in locm.values())),
        "per_run_localization": locm,
        "tilt_episode_test": tilt_test,
    }


def fmt(v, f="{:.1f}", none="n/a"):
    return none if v is None else f.format(v)


def to_markdown(res: dict) -> str:
    conds = list(res)
    L = ["## Navigation outcome", "",
         "| metric | " + " | ".join(conds) + " |",
         "|---|" + "---|" * len(conds)]
    L.append("| runs | " + " | ".join(str(res[c]["n_runs"]) for c in conds) + " |")
    L.append("| full route succeeded | " + " | ".join(
        f"{res[c]['full_route_success']}/{res[c]['n_runs']}" for c in conds) + " |")
    for leg in res[conds[0]]["leg_success"]:
        L.append(f"| {leg} | " + " | ".join(
            "{}/{}".format(*res[c]["leg_success"][leg]) for c in conds) + " |")
    L.append("| median recoveries per run | " + " | ".join(
        fmt(res[c]["recoveries_median"], "{:.0f}") for c in conds) + " |")
    L.append("| Nav2 false successes (total) | " + " | ".join(
        str(res[c]["false_successes_total"]) for c in conds) + " |")
    L.append("| median route time (sim s) | " + " | ".join(
        fmt(res[c]["route_sim_seconds_median"], "{:.0f}") for c in conds) + " |")

    L += ["", "## Localization (median across runs)", "",
          "| metric | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    rows = [
        ("runs with localization data", "localized_runs", "{:.0f}", ""),
        ("AMCL error, median", "amcl_median_cm", "{:.1f}", " cm"),
        ("AMCL error, p95", "amcl_p95_cm", "{:.1f}", " cm"),
        ("AMCL error on rough patch", "amcl_rough_median_cm", "{:.1f}", " cm"),
        ("time on rough patch", "rough_exposure_s", "{:.1f}", " s"),
        ("tilt p95", "tilt_p95_deg", "{:.1f}", " deg"),
    ]
    for name, key, f, unit in rows:
        L.append(f"| {name} | " + " | ".join(
            (fmt(res[c][key], f) + unit) if res[c][key] is not None else "n/a" for c in conds) + " |")
    L.append(f"| runs with AMCL error > {LOST_CM:.0f} cm | " + " | ".join(
        f"{res[c]['runs_lost']}/{res[c]['localized_runs']}" for c in conds) + " |")

    L += ["", "## Tilt episodes vs flat floor (pooled per condition)", "",
          "| condition | episodes | controls | median change, tilt | median change, flat | difference [95% CI] | p |",
          "|---|---|---|---|---|---|---|"]
    for c in conds:
        t = res[c]["tilt_episode_test"]
        if not t or "median_difference_cm" not in t:
            L.append(f"| {c} | 0 | | | | | |")
            continue
        lo, hi = t["ci95_cm"]
        L.append(f"| {c} | {t['episodes']} | {t['controls']} | {t['episode_median_delta_cm']:+.1f} cm | "
                 f"{t['control_median_delta_cm']:+.1f} cm | {t['median_difference_cm']:+.1f} cm "
                 f"[{lo:+.1f}, {hi:+.1f}] | {t['p_value_one_sided']:.3f} |")
    L += ["", f"Runs: " + "; ".join(f"{c}: {', '.join(res[c]['runs'])}" for c in conds)]
    return "\n".join(L)


def figure(res: dict, path: pathlib.Path) -> None:
    plt = le._mpl()
    if plt is None:
        return
    conds = list(res)
    legs = list(res[conds[0]]["leg_success"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.8 / len(conds)
    x = np.arange(len(legs))
    for i, c in enumerate(conds):
        rate = [res[c]["leg_success"][l][0] / max(1, res[c]["leg_success"][l][1]) for l in legs]
        a1.bar(x + (i - (len(conds) - 1) / 2) * w, np.array(rate) * 100, w,
               label=f"{c} (n={res[c]['n_runs']})")
    a1.set_xticks(x)
    a1.set_xticklabels(legs, rotation=20)
    a1.set_ylabel("leg success [%]")
    a1.set_ylim(0, 105)
    a1.set_title("Route legs succeeded")
    a1.legend()
    a1.grid(alpha=0.3, axis="y")

    data, names = [], []
    for c in conds:
        for key, tag in (("amcl_median_cm", "median"), ("amcl_rough_median_cm", "rough patch")):
            v = [m[key] for m in res[c]["per_run_localization"].values() if key in m]
            if v:
                data.append(v)
                names.append(f"{c}\n{tag}")
    if data:
        a2.boxplot(data)
        for i, v in enumerate(data, start=1):
            a2.scatter(np.full(len(v), i), v, s=18, zorder=3)
        a2.set_xticks(range(1, len(names) + 1))
        a2.set_xticklabels(names)
    a2.set_ylabel("AMCL position error per run [cm]")
    a2.set_title("Localization error, one point per run")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--routes", default=str(REPO / "results" / "routes"))
    ap.add_argument("--localization", default=str(REPO / "results" / "localization"))
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    ap.add_argument("--conditions", nargs="*",
                    help="only these condition prefixes, e.g. v3d_baseline v3d_terrain")
    ap.add_argument("--out", default=str(REPO / "docs" / "results" / "ab"))
    args = ap.parse_args()

    routes = load_routes(pathlib.Path(args.routes), set(args.exclude))
    loc = load_localization(pathlib.Path(args.localization), set(routes))
    by_cond = {}
    for lab in sorted(routes):
        cond = condition_of(lab)
        if args.conditions and cond not in args.conditions:
            continue
        by_cond.setdefault(cond, []).append(lab)
    if not by_cond:
        sys.exit("no runs match; check --conditions")
    order = sorted(by_cond, key=lambda c: ("baseline" not in c, c))
    res = {c: summarize_condition(by_cond[c], routes, loc) for c in order}

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(res)
    (out / "ab_summary.md").write_text(md + "\n")
    (out / "ab_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(md)
    print(f"\nexcluded: {', '.join(args.exclude) or 'none'}")
    figure(res, out / "ab_figure.png")


if __name__ == "__main__":
    main()
