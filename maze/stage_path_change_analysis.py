#!/usr/bin/env python3
"""
Analyse cached valuation dumps to compare start-goal values for pairs whose
optimal shortest-path changes after the incremental maze update vs those that
remain unchanged. Computes separate statistics for the first and second
generative replay stages and plots a bar chart with significance annotations.

Usage:
    python stage_path_change_analysis.py \
        --cache-dir ../logs/pretrain_10000/valuation_cache \
        --mode top_with_mmr \
        --output stage_path_change_bias.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

GridCoord = Tuple[int, int]
PathSet = frozenset[GridCoord]


def load_cache(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_pre_post(data: Dict[str, object], fname: str) -> Tuple[np.ndarray, np.ndarray]:
    pre_raw = data.get("pre_incremental_maze")
    post_raw = data.get("post_incremental_maze")
    candidate_raw = data.get("candidate_maze")
    prev_raw = data.get("prev_maze")

    pre_arr: np.ndarray | None = None
    post_arr: np.ndarray | None = None

    if pre_raw is not None:
        pre_arr = np.asarray(pre_raw, dtype=int)
        if post_raw is None:
            raise ValueError(f"Missing post_incremental_maze in {fname}")
        post_arr = np.asarray(post_raw, dtype=int)
    else:
        candidate_arr = np.asarray(candidate_raw, dtype=int) if candidate_raw is not None else None
        if post_raw is not None and candidate_arr is not None:
            post_candidate = np.asarray(post_raw, dtype=int)
            if not np.array_equal(post_candidate, candidate_arr):
                pre_arr = post_candidate
                post_arr = candidate_arr
        if pre_arr is None and prev_raw is not None and candidate_arr is not None:
            prev_arr = np.asarray(prev_raw, dtype=int)
            if not np.array_equal(prev_arr, candidate_arr):
                pre_arr = prev_arr
                post_arr = candidate_arr

    if pre_arr is None or post_arr is None:
        raise ValueError(
            f"Could not resolve before/after mazes for {fname}; "
            "ensure the cache includes baseline and updated layouts."
        )
    return pre_arr, post_arr


def _stage_label(stage: int | str, custom_label: str | None = None) -> str:
    if custom_label:
        return str(custom_label)
    try:
        stage_int = int(stage)
    except (ValueError, TypeError):
        return f"Stage {stage}"
    return "Aggregate" if stage_int == 0 else f"Stage {stage_int}"


def _mode_prefixes(mode: str) -> List[str] | None:
    if mode.lower() == "all":
        return None
    prefixes = [m.strip() for m in mode.split(",") if m.strip()]
    return prefixes or [mode]


def _stage_file_list(cache_dir: str, mode: str, stage: int) -> List[str]:
    entries = sorted(os.listdir(cache_dir))
    prefixes = _mode_prefixes(mode)
    matches: List[str] = []
    if stage == 0:
        for name in entries:
            if not name.endswith(".json") or "_stage" in name:
                continue
            if prefixes is None or any(name.endswith(f"{pref}.json") for pref in prefixes):
                matches.append(name)
        return matches

    suffix = f"_stage{stage}.json"
    for name in entries:
        if not name.endswith(suffix):
            continue
        if prefixes is None or any(name.endswith(f"{pref}_stage{stage}.json") for pref in prefixes):
            matches.append(name)
    return matches


def available_stages(cache_dir: str, mode: str) -> List[int]:
    entries = os.listdir(cache_dir)
    prefixes = _mode_prefixes(mode)
    stages: set[int] = set()
    for name in entries:
        if not name.endswith(".json"):
            continue
        if prefixes is not None and not any(
            name.endswith(f"{pref}.json") or f"_{pref}_stage" in name for pref in prefixes
        ):
            continue
        match = re.search(r"_stage(\d+)\.json$", name)
        if match:
            stages.add(int(match.group(1)))
        elif prefixes is None or "_stage" not in name:
            stages.add(0)
    return sorted(stages)


def neighbors(cell: GridCoord, grid: int) -> Iterable[GridCoord]:
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid and 0 <= nc < grid:
            yield nr, nc


def all_shortest_paths(maze: np.ndarray, grid: int, start_idx: int, goal_idx: int) -> List[PathSet]:
    sr, sc = divmod(int(start_idx), grid)
    gr, gc = divmod(int(goal_idx), grid)
    if maze[sr, sc] == 1 or maze[gr, gc] == 1:
        return []

    start = (sr, sc)
    goal = (gr, gc)
    dist: Dict[GridCoord, int] = {start: 0}
    q: deque[GridCoord] = deque([start])

    while q:
        cell = q.popleft()
        for nb in neighbors(cell, grid):
            nr, nc = nb
            if maze[nr, nc] == 1:
                continue
            if nb not in dist:
                dist[nb] = dist[cell] + 1
                q.append(nb)

    if goal not in dist:
        return []

    paths: List[PathSet] = []
    stack: List[GridCoord] = [start]

    def dfs(cell: GridCoord) -> None:
        if cell == goal:
            paths.append(frozenset(stack))
            return
        for nb in neighbors(cell, grid):
            nr, nc = nb
            if maze[nr, nc] == 1:
                continue
            if nb in dist and dist[nb] == dist[cell] + 1:
                stack.append(nb)
                dfs(nb)
                stack.pop()

    dfs(start)
    return paths


def shortest_path_length(maze: np.ndarray, start_idx: int, goal_idx: int) -> int | None:
    grid = int(maze.shape[0])
    sr, sc = divmod(int(start_idx), grid)
    gr, gc = divmod(int(goal_idx), grid)
    if maze[sr, sc] == 1 or maze[gr, gc] == 1:
        return None

    start = (sr, sc)
    goal = (gr, gc)
    dist: Dict[GridCoord, int] = {start: 0}
    q: deque[GridCoord] = deque([start])

    while q:
        cell = q.popleft()
        if cell == goal:
            return dist[cell]
        for nb in neighbors(cell, grid):
            nr, nc = nb
            if maze[nr, nc] == 1:
                continue
            if nb not in dist:
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return None


def mean_jaccard(prev_paths: Sequence[PathSet], post_paths: Sequence[PathSet]) -> float | None:
    if not prev_paths or not post_paths:
        return None
    total = 0.0
    count = 0
    for a in prev_paths:
        for b in post_paths:
            union = a | b
            if not union:
                dist = 0.0
            else:
                inter = a & b
                dist = 1.0 - len(inter) / len(union)
            total += dist
            count += 1
    if count == 0:
        return None
    return total / count


def collect_stage_stats(
    cache_dir: str,
    mode: str,
    stage: int,
    radius: int,
    path_length: int | None = None,
) -> Dict[str, np.ndarray]:
    changed_vals: List[float] = []
    unchanged_vals: List[float] = []
    near_cells: List[float] = []
    far_cells: List[float] = []
    total_sum = 0.0
    total_count = 0

    for fname in _stage_file_list(cache_dir, mode, stage):
        data = load_cache(os.path.join(cache_dir, fname))
        pre, post = _resolve_pre_post(data, fname)
        grid = int(pre.shape[0])
        pairs = data["all_pairs"]
        values = data["all_pair_values"]
        # For the unfiltered case, treat all values as contributing to the global mean.
        # When a specific path_length is requested, we instead accumulate the global
        # mean over only those pairs that satisfy the length constraint (see below).
        if path_length is None:
            total_sum += float(np.sum(values))
            total_count += len(values)

        # Identify changed cells (both added and removed)
        changed_cells = [(r, c) for r in range(grid) for c in range(grid) if pre[r, c] != post[r, c]]

        val_sum = np.zeros((grid, grid), dtype=float)
        val_cnt = np.zeros((grid, grid), dtype=float)

        for (s, g), val in zip(pairs, values):
            # Optionally restrict to a specific shortest-path length in the *post* maze
            # so that near/far comparisons are not confounded by path-length differences.
            if path_length is not None:
                pl = shortest_path_length(post, s, g)
                if pl is None or pl != path_length:
                    continue
                # For the filtered case, build the global mean from the same subset.
                total_sum += float(val)
                total_count += 1.0

            prev_paths = all_shortest_paths(pre, grid, s, g)
            post_paths = all_shortest_paths(post, grid, s, g)
            mean_dist = mean_jaccard(prev_paths, post_paths)
            if mean_dist is None:
                continue
            if mean_dist > 1e-9:
                changed_vals.append(float(val))
            else:
                unchanged_vals.append(float(val))

            # accumulate per-cell values
            rs, cs = divmod(int(s), grid)
            rg, cg = divmod(int(g), grid)
            val_sum[rs, cs] += float(val)
            val_cnt[rs, cs] += 1.0
            val_sum[rg, cg] += float(val)
            val_cnt[rg, cg] += 1.0

        if changed_cells:
            cell_means = np.full_like(val_sum, np.nan)
            mask = val_cnt > 0
            cell_means[mask] = val_sum[mask] / val_cnt[mask]
            for r in range(grid):
                for c in range(grid):
                    if not np.isfinite(cell_means[r, c]):
                        continue
                    dist = min(abs(r - rc[0]) + abs(c - rc[1]) for rc in changed_cells)
                    if dist <= radius:
                        near_cells.append(float(cell_means[r, c]))
                    else:
                        far_cells.append(float(cell_means[r, c]))

    return {
        "stage": stage,
        "stage_label": None,
        "changed": np.asarray(changed_vals, dtype=float),
        "unchanged": np.asarray(unchanged_vals, dtype=float),
        "near_cells": np.asarray(near_cells, dtype=float),
        "far_cells": np.asarray(far_cells, dtype=float),
        "global_sum": total_sum,
        "global_count": total_count,
    }


def collect_length_value_stats(
    cache_dir: str,
    mode: str,
    stage: int,
) -> Dict[str, object]:
    per_length: defaultdict[int, List[float]] = defaultdict(list)
    lengths: List[int] = []
    values: List[float] = []

    for fname in _stage_file_list(cache_dir, mode, stage):
        data = load_cache(os.path.join(cache_dir, fname))
        _, maze = _resolve_pre_post(data, fname)
        pairs = data["all_pairs"]
        val_list = data["all_pair_values"]
        for (start_idx, goal_idx), val in zip(pairs, val_list):
            path_len = shortest_path_length(maze, start_idx, goal_idx)
            if path_len is None:
                continue
            per_length[int(path_len)].append(float(val))
            lengths.append(int(path_len))
            values.append(float(val))

    length_arrays: Dict[int, np.ndarray] = {
        length: np.asarray(vals, dtype=float) for length, vals in per_length.items()
    }
    lengths_arr = np.asarray(lengths, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if lengths_arr.size > 1:
        corr = float(np.corrcoef(lengths_arr, values_arr)[0, 1])
    else:
        corr = float("nan")

    value_mean = float(values_arr.mean()) if values_arr.size else float("nan")
    return {
        "stage": stage,
        "stage_label": None,
        "per_length": length_arrays,
        "lengths": lengths_arr,
        "values": values_arr,
        "corr": corr,
        "value_mean": value_mean,
    }


def collect_manhattan_value_stats(
    cache_dir: str,
    mode: str,
    stage: int,
) -> Dict[str, object]:
    per_distance: defaultdict[int, List[float]] = defaultdict(list)
    distances: List[int] = []
    values: List[float] = []

    for fname in _stage_file_list(cache_dir, mode, stage):
        data = load_cache(os.path.join(cache_dir, fname))
        _, maze = _resolve_pre_post(data, fname)
        grid = int(maze.shape[0])
        pairs = data["all_pairs"]
        val_list = data["all_pair_values"]
        for (start_idx, goal_idx), val in zip(pairs, val_list):
            sr, sc = divmod(int(start_idx), grid)
            gr, gc = divmod(int(goal_idx), grid)
            dist = abs(sr - gr) + abs(sc - gc)
            per_distance[int(dist)].append(float(val))
            distances.append(int(dist))
            values.append(float(val))

    distance_arrays: Dict[int, np.ndarray] = {
        dist: np.asarray(vals, dtype=float) for dist, vals in per_distance.items()
    }
    distances_arr = np.asarray(distances, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if distances_arr.size > 1:
        corr = float(np.corrcoef(distances_arr, values_arr)[0, 1])
    else:
        corr = float("nan")

    value_mean = float(values_arr.mean()) if values_arr.size else float("nan")
    return {
        "stage": stage,
        "stage_label": None,
        "per_distance": distance_arrays,
        "distances": distances_arr,
        "values": values_arr,
        "corr": corr,
        "value_mean": value_mean,
    }


def summarise(stats_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    changed = stats_dict["changed"]
    unchanged = stats_dict["unchanged"]
    total_count = stats_dict["global_count"]
    total_sum = stats_dict["global_sum"]
    global_mean = float(total_sum / total_count) if total_count else float("nan")

    changed_mean = float(changed.mean()) if changed.size else float("nan")
    unchanged_mean = float(unchanged.mean()) if unchanged.size else float("nan")
    changed_diff = changed_mean - global_mean if np.isfinite(changed_mean) and np.isfinite(global_mean) else float("nan")
    unchanged_diff = unchanged_mean - global_mean if np.isfinite(unchanged_mean) and np.isfinite(global_mean) else float("nan")

    changed_sem = float(changed.std(ddof=1) / np.sqrt(changed.size)) if changed.size > 1 else float("nan")
    unchanged_sem = float(unchanged.std(ddof=1) / np.sqrt(unchanged.size)) if unchanged.size > 1 else float("nan")

    if changed.size > 1 and unchanged.size > 1:
        t_stat, p_val = ttest_ind(changed, unchanged, equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")

    near = stats_dict["near_cells"]
    far = stats_dict["far_cells"]
    near_mean = float(near.mean()) if near.size else float("nan")
    far_mean = float(far.mean()) if far.size else float("nan")
    near_diff = near_mean - global_mean if np.isfinite(near_mean) and np.isfinite(global_mean) else float("nan")
    far_diff = far_mean - global_mean if np.isfinite(far_mean) and np.isfinite(global_mean) else float("nan")
    near_sem = float(near.std(ddof=1) / np.sqrt(near.size)) if near.size > 1 else float("nan")
    far_sem = float(far.std(ddof=1) / np.sqrt(far.size)) if far.size > 1 else float("nan")

    if near.size > 1 and far.size > 1:
        t_nf, p_nf = ttest_ind(near, far, equal_var=False)
    else:
        t_nf, p_nf = float("nan"), float("nan")

    return {
        "stage": stats_dict.get("stage"),
        "stage_label": stats_dict.get("stage_label"),
        "global_mean": global_mean,
        "changed_mean": changed_mean,
        "unchanged_mean": unchanged_mean,
        "changed_diff": changed_diff,
        "unchanged_diff": unchanged_diff,
        "changed_sem": changed_sem,
        "unchanged_sem": unchanged_sem,
        "changed_count": int(changed.size),
        "unchanged_count": int(unchanged.size),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "near_diff": near_diff,
        "far_diff": far_diff,
        "near_mean": near_mean,
        "far_mean": far_mean,
        "near_sem": near_sem,
        "far_sem": far_sem,
        "near_count": int(near.size),
        "far_count": int(far.size),
        "p_near_far": float(p_nf),
    }


def _concat_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    valid = [arr for arr in arrays if isinstance(arr, np.ndarray) and arr.size]
    if not valid:
        return np.asarray([], dtype=float)
    return np.concatenate(valid)


def _merge_stage_stats_dicts(
    dicts: Sequence[Dict[str, np.ndarray]],
    stage_label: str,
    stage_value: int,
) -> Dict[str, np.ndarray]:
    if not dicts:
        raise ValueError("No stage statistics to merge.")

    merged = {
        "stage": stage_value,
        "stage_label": stage_label,
        "changed": _concat_arrays([d["changed"] for d in dicts]),
        "unchanged": _concat_arrays([d["unchanged"] for d in dicts]),
        "near_cells": _concat_arrays([d["near_cells"] for d in dicts]),
        "far_cells": _concat_arrays([d["far_cells"] for d in dicts]),
        "global_sum": float(sum(d.get("global_sum", 0.0) for d in dicts)),
        "global_count": float(sum(d.get("global_count", 0.0) for d in dicts)),
    }
    return merged


def _merge_length_stats_dicts(
    dicts: Sequence[Dict[str, object]],
    stage_label: str,
    stage_value: int,
) -> Dict[str, object]:
    if not dicts:
        raise ValueError("No length statistics to merge.")

    from collections import defaultdict

    per_length: defaultdict[int, List[np.ndarray]] = defaultdict(list)
    for stats in dicts:
        for length, arr in stats.get("per_length", {}).items():  # type: ignore[assignment]
            per_length[int(length)].append(np.asarray(arr, dtype=float))

    merged_per_length = {
        length: _concat_arrays(arr_list) for length, arr_list in per_length.items()
    }
    lengths_arr = _concat_arrays([np.asarray(stats.get("lengths"), dtype=float) for stats in dicts])
    values_arr = _concat_arrays([np.asarray(stats.get("values"), dtype=float) for stats in dicts])
    if lengths_arr.size > 1 and values_arr.size == lengths_arr.size:
        corr = float(np.corrcoef(lengths_arr, values_arr)[0, 1])
    else:
        corr = float("nan")
    value_mean = float(values_arr.mean()) if values_arr.size else float("nan")
    return {
        "stage": stage_value,
        "stage_label": stage_label,
        "per_length": merged_per_length,
        "lengths": lengths_arr,
        "values": values_arr,
        "corr": corr,
        "value_mean": value_mean,
    }


def _merge_manhattan_stats_dicts(
    dicts: Sequence[Dict[str, object]],
    stage_label: str,
    stage_value: int,
) -> Dict[str, object]:
    if not dicts:
        raise ValueError("No Manhattan statistics to merge.")

    from collections import defaultdict

    per_distance: defaultdict[int, List[np.ndarray]] = defaultdict(list)
    for stats in dicts:
        for dist, arr in stats.get("per_distance", {}).items():  # type: ignore[assignment]
            per_distance[int(dist)].append(np.asarray(arr, dtype=float))

    merged_per_distance = {
        dist: _concat_arrays(arr_list) for dist, arr_list in per_distance.items()
    }
    distances_arr = _concat_arrays([np.asarray(stats.get("distances"), dtype=float) for stats in dicts])
    values_arr = _concat_arrays([np.asarray(stats.get("values"), dtype=float) for stats in dicts])
    if distances_arr.size > 1 and values_arr.size == distances_arr.size:
        corr = float(np.corrcoef(distances_arr, values_arr)[0, 1])
    else:
        corr = float("nan")
    value_mean = float(values_arr.mean()) if values_arr.size else float("nan")
    return {
        "stage": stage_value,
        "stage_label": stage_label,
        "per_distance": merged_per_distance,
        "distances": distances_arr,
        "values": values_arr,
        "corr": corr,
        "value_mean": value_mean,
    }


def _sig_label(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _plot_changed_vs_unchanged(ax: plt.Axes, res: Dict[str, float], *, show_ylabel: bool = False) -> None:
    changed_count = int(res.get("changed_count", 0))
    unchanged_count = int(res.get("unchanged_count", 0))
    stage_name = _stage_label(res.get("stage", "?"), res.get("stage_label"))
    if changed_count == 0 and unchanged_count == 0:
        ax.set_title(stage_name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    means = [res["changed_diff"], res["unchanged_diff"]]
    sems = [res["changed_sem"], res["unchanged_sem"]]
    labels = ["Changed", "Unchanged"]
    x = np.arange(len(labels))

    bar_heights = np.nan_to_num(np.asarray(means, dtype=float), nan=0.0)
    error_bars = np.nan_to_num(np.asarray(sems, dtype=float), nan=0.0)

    ax.bar(x, bar_heights, yerr=error_bars, color="tab:blue", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(stage_name)
    if show_ylabel:
        ax.set_ylabel("Mean-centered value")
    ax.axhline(0.0, color="black", linewidth=1)

    valid_extents = [
        (
            bar_heights[i] - (error_bars[i] if np.isfinite(error_bars[i]) else 0.0),
            bar_heights[i] + (error_bars[i] if np.isfinite(error_bars[i]) else 0.0),
        )
        for i in range(len(bar_heights))
    ]
    finite_bottoms = [lo for lo, hi in valid_extents if np.isfinite(lo)]
    finite_tops = [hi for lo, hi in valid_extents if np.isfinite(hi)]
    if finite_bottoms and finite_tops:
        ymin = min(finite_bottoms + [0.0])
        ymax = max(finite_tops + [0.0])
    else:
        ymin = -1.0
        ymax = 1.0

    ymax = max(ymax, 0.0)
    ymin = min(ymin, 0.0)
    vertical_span = ymax - ymin if ymax > ymin else 1e-3
    increment = vertical_span * 0.25
    yline = ymax + increment * 0.5
    ax.text(
        0.5,
        yline + increment * 0.15,
        _sig_label(res.get("p_value", float("nan"))),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.plot([0, 1], [yline, yline], color="black", linewidth=1)
    ax.set_ylim(bottom=ymin - increment * 0.6, top=yline + increment * 0.7)


def _plot_length_curve(
    ax: plt.Axes,
    res: Dict[str, object],
    *,
    show_ylabel: bool = False,
    xlabel: str = "Shortest path length",
) -> None:
    per_length: Dict[int, np.ndarray] = res.get("per_length", {})  # type: ignore[assignment]
    stage = res.get("stage", "?")
    stage_name = _stage_label(stage, res.get("stage_label"))
    if not per_length:
        ax.set_title(stage_name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    sorted_lengths = sorted(per_length.keys())
    means = [float(per_length[length].mean()) for length in sorted_lengths]
    sems: List[float] = []
    for length in sorted_lengths:
        arr = per_length[length]
        if arr.size > 1:
            sems.append(float(arr.std(ddof=1) / np.sqrt(arr.size)))
        elif arr.size == 1:
            sems.append(0.0)
        else:
            sems.append(float("nan"))

    finite_indices = [i for i, val in enumerate(means) if np.isfinite(val)]
    if not finite_indices:
        ax.set_title(stage_name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.errorbar(
        sorted_lengths,
        means,
        yerr=sems,
        fmt="-o",
        color="tab:blue",
        capsize=4,
        markersize=3,
        linewidth=1.2,
    )

    if sorted_lengths:
        first = max(2, sorted_lengths[0] + (sorted_lengths[0] % 2))
        last = sorted_lengths[-1]
        ticks = list(range(first, last + 1, 2))
        if last % 2 != 0 and (not ticks or ticks[-1] != last):
            ticks.append(last)
        if not ticks:
            ticks = sorted_lengths
        ax.set_xticks(ticks)
    else:
        ax.set_xticks(sorted_lengths)

    ax.set_title(stage_name)
    if show_ylabel:
        ax.set_ylabel("Mean-centered value")
    ax.set_xlabel(xlabel)
    ax.axhline(0.0, color="black", linewidth=1)

    finite_means = [means[i] for i in finite_indices]
    finite_sems = [sems[i] if np.isfinite(sems[i]) else 0.0 for i in finite_indices]
    ymax = max(finite_means[i] + finite_sems[i] for i in range(len(finite_means)))
    ymin = min(finite_means[i] - finite_sems[i] for i in range(len(finite_means)))
    if ymax <= ymin:
        pad = 1e-3
    else:
        pad = (ymax - ymin) * 0.25
    ax.set_ylim(ymin - pad * 0.6, ymax + pad * 0.7)

def plot_results(results: Sequence[Dict[str, float]], output: str) -> None:
    if not results:
        print("No stage summaries to plot; skipped changed/unchanged bar chart.")
        return

    cols = len(results)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 3), sharey=cols > 1)
    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(np.asarray(axes).reshape(-1))

    for idx, res in enumerate(results):
        _plot_changed_vs_unchanged(axes_list[idx], res, show_ylabel=(idx == 0))

    for idx in range(len(results), len(axes_list)):
        axes_list[idx].set_axis_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved bar chart to {output}")


def plot_near_far(results: Sequence[Dict[str, float]], output: str) -> None:
    if not results:
        print("No stage summaries to plot; skipped near/far analysis.")
        return

    cols = len(results)
    # Use a narrower figure than the main bar plots to keep the near/far panel compact.
    fig, axes = plt.subplots(1, cols, figsize=(1.7 * cols, 2.9), sharey=cols > 1)
    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(np.asarray(axes).reshape(-1))

    for idx, res in enumerate(results):
        ax = axes_list[idx]
        near_count = int(res.get("near_count", 0))
        far_count = int(res.get("far_count", 0))
        stage_name = _stage_label(res.get("stage", "?"), res.get("stage_label"))
        if near_count == 0 and far_count == 0:
            ax.set_title(stage_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Plot raw mean values for near/far cells (no global-mean subtraction)
        means = [res["near_mean"], res["far_mean"]]
        sems = [res["near_sem"], res["far_sem"]]
        labels = ["Near", "Far"]
        x = np.arange(len(labels))
        bar_heights = np.nan_to_num(np.asarray(means, dtype=float), nan=0.0)
        error_bars = np.nan_to_num(np.asarray(sems, dtype=float), nan=0.0)
        ax.bar(x, bar_heights, yerr=error_bars, color="tab:blue", capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(stage_name)
        if idx == 0:
            ax.set_ylabel("Mean value")
        ax.axhline(0.0, color="black", linewidth=1)
        valid_extents = [
            (
                bar_heights[i] - (error_bars[i] if np.isfinite(error_bars[i]) else 0.0),
                bar_heights[i] + (error_bars[i] if np.isfinite(error_bars[i]) else 0.0),
            )
            for i in range(len(bar_heights))
        ]
        finite_bottoms = [lo for lo, hi in valid_extents if np.isfinite(lo)]
        finite_tops = [hi for lo, hi in valid_extents if np.isfinite(hi)]
        # Fix the vertical axis to [0, 7e-4] so that value=0 is the x-axis and
        # the top of the plot is consistently 0.0007, preventing the sig label
        # from colliding with the subplot title.
        ymin = 0.0
        ymax = 7e-4
        vertical_span = ymax - ymin
        margin = vertical_span * 0.1
        # Place the sig line somewhat below the top limit
        yline = ymax - margin * 1.5
        # Slightly raise the significance bar for better visual separation
        yline += 3e-5
        ax.text(
            0.5,
            yline + margin * 0.1,
            _sig_label(res.get("p_near_far", float("nan"))),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.plot([0, 1], [yline, yline], color="black", linewidth=1)
        ax.set_ylim(bottom=ymin, top=ymax)

    for idx in range(len(results), len(axes_list)):
        axes_list[idx].set_axis_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved near/far bar chart to {output}")


def plot_value_by_length(length_results: Sequence[Dict[str, object]], output: str) -> None:
    if not length_results:
        print("No path-length statistics to plot; skipped path length figure.")
        return

    cols = len(length_results)
    fig, axes = plt.subplots(1, cols, figsize=(3.8 * cols, 2.9), sharey=cols > 1)
    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(np.asarray(axes).reshape(-1))

    for idx, res in enumerate(length_results):
        _plot_length_curve(axes_list[idx], res, show_ylabel=(idx == 0), xlabel="Shortest path length")

    for idx in range(len(length_results), len(axes_list)):
        axes_list[idx].set_axis_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    pdf_output = os.path.splitext(output)[0] + ".pdf"
    fig.savefig(pdf_output)
    plt.close(fig)
    print(f"Saved path length plot to {output} and {pdf_output}")


def plot_value_by_manhattan(distance_results: Sequence[Dict[str, object]], output: str) -> None:
    if not distance_results:
        print("No Manhattan-distance statistics to plot; skipped Manhattan figure.")
        return

    cols = len(distance_results)
    fig, axes = plt.subplots(1, cols, figsize=(3.8 * cols, 2.9), sharey=cols > 1)
    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(np.asarray(axes).reshape(-1))

    for idx, res in enumerate(distance_results):
        # Reuse plotting helper by adapting dictionary structure to expected keys
        transformed = {
            "stage": res.get("stage"),
            "stage_label": res.get("stage_label"),
            "per_length": res.get("per_distance", {}),
        }
        _plot_length_curve(
            axes_list[idx],
            transformed,
            show_ylabel=(idx == 0),
            xlabel="Manhattan distance",
        )

    for idx in range(len(distance_results), len(axes_list)):
        axes_list[idx].set_axis_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    pdf_output = os.path.splitext(output)[0] + ".pdf"
    fig.savefig(pdf_output)
    plt.close(fig)
    print(f"Saved Manhattan-distance plot to {output} and {pdf_output}")


def plot_combined(
    results: Sequence[Dict[str, float]],
    manhattan_results: Sequence[Dict[str, object]],
    output: str,
) -> None:
    """Create a four-panel summary figure:
    - Left two panels: changed vs unchanged bar plots for the first two stages.
    - Right two panels: value vs Manhattan distance curves for the same stages.
    """
    if len(results) < 2 or len(manhattan_results) < 2:
        print("Skipping combined plot: need at least two stages with both summaries and Manhattan stats.")
        return

    used_results = list(results[:2])
    used_manhattan_results = list(manhattan_results[:2])
    if len(results) > 2 or len(manhattan_results) > 2:
        print("Combined plot uses the first two stages; additional stages are omitted.")

    cols = len(used_results) + len(used_manhattan_results)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 3.0))
    axes_list = list(np.asarray(axes).reshape(-1))

    for idx, res in enumerate(used_results):
        _plot_changed_vs_unchanged(axes_list[idx], res, show_ylabel=(idx == 0))

    offset = len(used_results)
    for idx, res in enumerate(used_manhattan_results):
        # Adapt Manhattan stats to the structure expected by _plot_length_curve
        transformed = {
            "stage": res.get("stage"),
            "stage_label": res.get("stage_label"),
            "per_length": res.get("per_distance", {}),
        }
        _plot_length_curve(
            axes_list[offset + idx],
            transformed,
            show_ylabel=(idx == 0),
            xlabel="Manhattan distance",
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    pdf_output = os.path.splitext(output)[0] + ".pdf"
    fig.savefig(pdf_output)
    plt.close(fig)
    print(f"Saved combined plot to {output} and {pdf_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "logs", "pretrain_10000", "valuation_cache"),
        help="Directory containing valuation cache JSON files.",
    )
    parser.add_argument(
        "--mode",
        default="top_with_mmr",
        help="Mode name(s) to include (e.g., 'top_with_mmr', 'longest_paths', or 'all' / comma-separated).",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage numbers to include (use 0 for baseline files without '_stage').",
    )
    parser.add_argument(
        "--last-n-stages",
        type=int,
        default=None,
        help="Automatically select the last N available stages (ignores aggregated stage 0).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "stage_path_change_bias.png"),
        help="Path to save the bar chart.",
    )
    parser.add_argument(
        "--length-output",
        default=os.path.join(os.path.dirname(__file__), "stage_path_length_values.png"),
        help="Path to save the path length vs value plot.",
    )
    parser.add_argument(
        "--manhattan-output",
        default=os.path.join(os.path.dirname(__file__), "stage_path_manhattan_values.png"),
        help="Path to save the Manhattan distance vs value plot.",
    )
    parser.add_argument(
        "--combined-output",
        default=os.path.join(os.path.dirname(__file__), "stage_path_value_summary.png"),
        help="Path to save the combined four-panel figure.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Manhattan radius used to define 'near' cells (default: 2).",
    )
    parser.add_argument(
        "--path-length",
        type=int,
        default=6,
        help="Restrict near/far analysis to pairs whose shortest path in the post-update maze has this length (default: 6).",
    )
    parser.add_argument(
        "--aggregate-steps",
        action="store_true",
        help="Aggregate odd-numbered stages as Step 1 and even-numbered stages as Step 2.",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    if args.stages and args.last_n_stages:
        raise SystemExit("Please specify either --stages or --last-n-stages, not both.")
    if args.last_n_stages is not None and args.last_n_stages <= 0:
        raise SystemExit("--last-n-stages must be a positive integer.")

    if args.stages:
        try:
            stage_list = sorted({int(s.strip()) for s in args.stages.split(",") if s.strip()})
        except ValueError as exc:
            raise SystemExit(f"Invalid stage specification: {exc}") from exc
        if not stage_list:
            raise SystemExit("No valid stage numbers found in --stages argument.")
    else:
        if args.last_n_stages is not None:
            detected = [stage for stage in available_stages(cache_dir, args.mode) if stage != 0]
            if not detected:
                raise SystemExit(
                    f"No staged valuation files found for mode '{args.mode}' in {cache_dir}."
                )
            if args.last_n_stages < len(detected):
                stage_list = detected[-args.last_n_stages :]
            else:
                stage_list = detected
        else:
            aggregated_available = _stage_file_list(cache_dir, args.mode, 0)
            if aggregated_available:
                stage_list = [0]
            else:
                stage_list = [stage for stage in (1, 2) if _stage_file_list(cache_dir, args.mode, stage)]
                if not stage_list:
                    detected = available_stages(cache_dir, args.mode)
                    if detected:
                        stage_list = detected[-2:] if len(detected) > 2 else detected
    if not stage_list:
        raise SystemExit(f"No valuation stage files found for mode '{args.mode}' in {cache_dir}.")

    stage_list = sorted(stage_list)

    # We collect two variants of the stage statistics:
    # - stage_entries_all: uses all pairs (no path-length filtering) and feeds the
    #   main changed/unchanged and length/Manhattan plots.
    # - stage_entries_near_far: optionally uses a fixed path length and is used
    #   only for the near vs far cell analysis.
    stage_entries_all: List[Dict[str, object]] = []
    stage_entries_near_far: List[Dict[str, object]] = []
    for stage in stage_list:
        # Unfiltered stats for global / changed-vs-unchanged analysis
        stage_stats_all = collect_stage_stats(cache_dir, args.mode, stage, args.radius, path_length=None)
        length_stats = collect_length_value_stats(cache_dir, args.mode, stage)
        manhattan_stats = collect_manhattan_value_stats(cache_dir, args.mode, stage)
        stage_entries_all.append(
            {
                "stage": stage,
                "stage_label": None,
                "stats": stage_stats_all,
                "length": length_stats,
                "manhattan": manhattan_stats,
            }
        )

        # Optionally, a length-filtered copy used only for near/far plots
        if args.path_length is not None:
            stage_stats_nf = collect_stage_stats(cache_dir, args.mode, stage, args.radius, args.path_length)
            stage_entries_near_far.append(
                {
                    "stage": stage,
                    "stage_label": None,
                    "stats": stage_stats_nf,
                    "length": length_stats,
                    "manhattan": manhattan_stats,
                }
            )

    if args.aggregate_steps:
        aggregated_all: List[Dict[str, object]] = []
        aggregated_nf: List[Dict[str, object]] = []

        def _aggregate(entries: List[Dict[str, object]], label: str, stage_value: int, parity: int):
            selected = [
                entry
                for entry in entries
                if isinstance(entry["stage"], int) and int(entry["stage"]) % 2 == parity
            ]
            if not selected:
                return None
            merged_stats = _merge_stage_stats_dicts([entry["stats"] for entry in selected], label, stage_value)  # type: ignore[arg-type]
            merged_length = _merge_length_stats_dicts([entry["length"] for entry in selected], label, stage_value)  # type: ignore[arg-type]
            merged_manhattan = _merge_manhattan_stats_dicts([entry["manhattan"] for entry in selected], label, stage_value)  # type: ignore[arg-type]
            return {
                "stage": stage_value,
                "stage_label": label,
                "stats": merged_stats,
                "length": merged_length,
                "manhattan": merged_manhattan,
            }

        # Aggregate for the unfiltered stats (used for main plots)
        odd_all = _aggregate(stage_entries_all, "Step 1", 1, parity=1)
        even_all = _aggregate(stage_entries_all, "Step 2", 2, parity=0)
        if odd_all:
            aggregated_all.append(odd_all)
        if even_all:
            aggregated_all.append(even_all)
        stage_entries_all = aggregated_all

        # And, if present, aggregate the length-filtered stats for near/far only
        if stage_entries_near_far:
            odd_nf = _aggregate(stage_entries_near_far, "Step 1", 1, parity=1)
            even_nf = _aggregate(stage_entries_near_far, "Step 2", 2, parity=0)
            if odd_nf:
                aggregated_nf.append(odd_nf)
            if even_nf:
                aggregated_nf.append(even_nf)
            stage_entries_near_far = aggregated_nf

    analysis_labels = [
        _stage_label(entry["stage"], entry.get("stage_label"))  # type: ignore[arg-type]
        for entry in stage_entries_all
    ]
    print(f"Analysing: {', '.join(analysis_labels)}")

    results = []
    length_results: List[Dict[str, object]] = []
    manhattan_results: List[Dict[str, object]] = []

    for entry in stage_entries_all:
        summary = summarise(entry["stats"])  # type: ignore[arg-type]
        results.append(summary)

        length_stats = entry["length"]  # type: ignore[arg-type]
        length_results.append(length_stats)
        lengths_arr = length_stats.get("lengths")
        corr_val = float(length_stats.get("corr", float("nan")))
        count = int(lengths_arr.size) if isinstance(lengths_arr, np.ndarray) else 0

        manhattan_stats = entry["manhattan"]  # type: ignore[arg-type]
        manhattan_results.append(manhattan_stats)
        manhattan_arr = manhattan_stats.get("distances")
        manhattan_corr = float(manhattan_stats.get("corr", float("nan")))
        manhattan_count = int(manhattan_arr.size) if isinstance(manhattan_arr, np.ndarray) else 0

        stage_name = _stage_label(summary.get("stage", "?"), summary.get("stage_label"))
        print(
            f"{stage_name}: changed n={summary['changed_count']} mean={summary['changed_mean']:.4f} "
            f"unchanged n={summary['unchanged_count']} mean={summary['unchanged_mean']:.4f} "
            f"Welch t={summary['t_stat']:.3f} p={summary['p_value']:.3e}"
        )
        print(f"  Path length correlation: n={count} corr={corr_val:.4f}")
        print(f"  Manhattan distance correlation: n={manhattan_count} corr={manhattan_corr:.4f}")

    plot_results(results, os.path.abspath(args.output))
    near_far_output = os.path.splitext(os.path.abspath(args.output))[0] + "_near_far.png"
    # For near/far, prefer the optional length-filtered stats if provided;
    # otherwise, fall back to the unfiltered summaries.
    if stage_entries_near_far:
        nf_results = [summarise(entry["stats"]) for entry in stage_entries_near_far]  # type: ignore[arg-type]
        plot_near_far(nf_results, os.path.abspath(near_far_output))
    else:
        plot_near_far(results, os.path.abspath(near_far_output))
    if length_results:
        length_path = os.path.abspath(args.length_output)
        plot_value_by_length(length_results, length_path)
    if manhattan_results:
        manhattan_path = os.path.abspath(args.manhattan_output)
        plot_value_by_manhattan(manhattan_results, manhattan_path)
        combined_path = os.path.abspath(args.combined_output)
        plot_combined(results, manhattan_results, combined_path)


if __name__ == "__main__":
    main()
