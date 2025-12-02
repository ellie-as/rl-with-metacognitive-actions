#!/usr/bin/env python3
import os
import argparse
import random
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
try:
    from IPython import get_ipython
    if get_ipython() is None:
        matplotlib.use("Agg")
except Exception:
    # Fallback to non-interactive backend in headless environments
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from meta_env import MetaEnv
from seed_utils import seed_all


EPISODES_TO_AVG = 10


def _extract_episode_metrics(
    rewards: List[float],
    successes: List[float],
    actions_per_episode: int,
) -> Tuple[List[float], List[float]]:
    if actions_per_episode <= 0:
        return [], []
    episode_rewards: List[float] = []
    episode_success: List[float] = []
    for idx in range(actions_per_episode - 1, len(rewards), actions_per_episode):
        episode_rewards.append(float(rewards[idx]))
        if successes:
            episode_success.append(float(successes[idx]))
    return episode_rewards, episode_success


def _to_pair_list(pairs: Optional[List[Tuple[int, int]]]) -> List[List[int]]:
    if not pairs:
        return []
    return [[int(a), int(b)] for a, b in pairs]

def _maze_to_nested_list(maze) -> Optional[List[List[int]]]:
    if maze is None:
        return None
    try:
        return np.asarray(maze).astype(int).tolist()
    except Exception:
        try:
            return maze.tolist()
        except Exception:
            return None


def _collect_valuation_metadata(env: MetaEnv, stage: Optional[int] = None) -> Dict[str, Any]:
    all_pairs = getattr(env, "last_all_pairs", None)
    all_pair_values = getattr(env, "last_all_pair_values", None)
    selected_pairs = getattr(env, "last_selected_pairs", None)
    sampled_pairs = getattr(env, "last_sampled_pairs", None)
    sampled_values = getattr(env, "last_pair_values", None)
    val_history = getattr(env, "val_history", None)
    val_success_history = getattr(env, "val_success_history", None)
    pre_inc_maze = getattr(env, "_pre_incremental_maze", None)
    post_inc_maze = getattr(env, "_curr_maze", None)
    prev_maze = getattr(env, "_prev_maze", None)
    candidates_maze = getattr(env, "_candidates_maze_snapshot", None)
    if candidates_maze is None:
        candidates_maze = getattr(env, "_last_candidates_maze_snapshot", None)

    def _filter_pairs(pairs, values, maze):
        if pairs is None or maze is None:
            return pairs, values
        arr = np.array(maze)
        grid = arr.shape[0]
        filtered_pairs = []
        filtered_values = [] if values is not None else None
        for idx, pair in enumerate(pairs):
            s, g = pair
            sr, sc = divmod(int(s), grid)
            if arr[sr, sc] != 0:
                continue
            filtered_pairs.append(pair)
            if filtered_values is not None and idx < len(values):
                filtered_values.append(values[idx])
        if values is not None:
            return filtered_pairs, filtered_values
        return filtered_pairs, values

    filter_maze = candidates_maze if candidates_maze is not None else post_inc_maze
    if all_pairs:
        all_pairs, all_pair_values = _filter_pairs(all_pairs, all_pair_values, filter_maze)
    if selected_pairs:
        selected_pairs, _ = _filter_pairs(selected_pairs, None, filter_maze)
    if sampled_pairs:
        sampled_pairs, sampled_values = _filter_pairs(sampled_pairs, sampled_values, filter_maze)

    metadata = {
        "mode": getattr(env, "mode", None),
        "stage": int(stage) if stage is not None else None,
        "all_pairs": _to_pair_list(all_pairs),
        "all_pair_values": [float(v) for v in all_pair_values] if all_pair_values else [],
        "selected_pairs": _to_pair_list(selected_pairs),
        "sampled_pairs": _to_pair_list(sampled_pairs),
        "sampled_pair_values": [float(v) for v in sampled_values] if sampled_values else [],
        "val_history": [float(v) for v in val_history] if val_history else [],
        "val_success_history": [float(v) for v in val_success_history] if val_success_history else [],
        "pre_incremental_maze": _maze_to_nested_list(pre_inc_maze),
        "post_incremental_maze": _maze_to_nested_list(post_inc_maze),
        "prev_maze": _maze_to_nested_list(prev_maze),
        "candidate_maze": _maze_to_nested_list(candidates_maze),
        "initial_training_steps": int(getattr(Config, "INITIAL_TRAINING_STEPS", 0)),
        "episodes_per_burst": int(getattr(Config, "EPISODES_PER_BURST", 0)),
    }
    metadata["valued"] = bool(metadata["all_pair_values"])
    return metadata


def run_sequence(
    actions: List[int],
    seed: Optional[int] = None,
    initial_steps: Optional[int] = None,
    return_metadata: bool = False,
    return_success: bool = False,
) -> Any:
    if seed is None:
        seed = 1
    seed_all(seed)

    if initial_steps is None:
        initial_steps = getattr(Config, "INITIAL_TRAINING_STEPS", 0)
    env = MetaEnv(mode=Config.MODE, initial_training_steps=int(initial_steps), base_seed=seed)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)

    rewards = []
    success_history: List[float] = []
    stage_snapshots: List[Dict[str, Any]] = []
    stage_counter = 0
    for a in actions:
        obs, r, done, _, _ = env.step(int(a))
        rewards.append(float(r))
        success_history.append(float(getattr(env, "last_eval_success", 0.0)))
        if return_metadata and int(a) == 1:
            stage_counter += 1
            stage_snapshots.append(_collect_valuation_metadata(env, stage=stage_counter))
        if done:
            # keep running next actions in a new episode to accumulate comparable totals
            obs, _ = env.reset(seed=seed)
    metadata: Optional[Dict[str, Any]] = None
    if return_metadata:
        metadata = _collect_valuation_metadata(env)
        metadata["stages"] = stage_snapshots
        metadata["success_history"] = [float(x) for x in success_history]

    try:
        env.close()
    except Exception:
        pass

    if return_metadata and return_success:
        return rewards, success_history, metadata
    if return_metadata:
        return rewards, metadata
    if return_success:
        return rewards, success_history
    return rewards


def save_valuation_dump(
    base_logs_dir: str,
    steps: int,
    mode: str,
    seed: int,
    metadata: Optional[Dict[str, Any]],
    stage: Optional[int] = None,
    require_values: bool = True,
) -> Optional[str]:
    if not metadata:
        return None
    if require_values and not metadata.get("valued"):
        return None

    all_pairs = metadata.get("all_pairs", []) or []
    all_pair_values = metadata.get("all_pair_values", []) or []
    if require_values and (not all_pairs or not all_pair_values):
        return None

    selected_pairs = metadata.get("selected_pairs", [])
    sampled_pairs = metadata.get("sampled_pairs", [])
    sampled_values = metadata.get("sampled_pair_values", [])
    val_history = metadata.get("val_history", [])
    val_success_history = metadata.get("val_success_history", [])
    success_history = metadata.get("success_history", [])

    dump = {
        "seed": int(seed),
        "mode": mode,
        "stage": int(stage) if stage is not None else None,
        "initial_training_steps": int(steps),
        "lambda_param": float(getattr(Config, "LAMBDA_PARAM", 0.0)) if hasattr(Config, "LAMBDA_PARAM") else None,
        "episodes_per_burst": int(getattr(Config, "EPISODES_PER_BURST", 0)),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "all_pairs": all_pairs,
        "all_pair_values": all_pair_values,
        "selected_pairs": selected_pairs,
        "sampled_pairs": sampled_pairs,
        "sampled_pair_values": sampled_values,
        "val_history": val_history,
        "val_success_history": val_success_history,
        "success_history": success_history,
        "pre_incremental_maze": metadata.get("pre_incremental_maze"),
        "post_incremental_maze": metadata.get("post_incremental_maze"),
        "prev_maze": metadata.get("prev_maze"),
        "candidate_maze": metadata.get("candidate_maze"),
    }

    valuation_dir = os.path.join(base_logs_dir, f"pretrain_{steps}", "valuation_cache")
    os.makedirs(valuation_dir, exist_ok=True)
    suffix = f"_stage{int(stage)}" if stage is not None else ""
    out_path = os.path.join(valuation_dir, f"seed_{seed}_{mode}{suffix}.json")
    with open(out_path, "w") as fh:
        json.dump(dump, fh, indent=2)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Action selection baselines")
    parser.add_argument(
        "--skip-valuation",
        action="store_true",
        help="Skip data valuation section; only run learned vs memory replay baselines",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to run (starting at Config.SEED)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Common settings
    # Use the fully-covered debug world model so dream-environment training has reliable transitions.
    setattr(Config, "WORLD_MODEL_TYPE", "cache")
    setattr(Config, "SEED", 1)
    seeds = [int(Config.SEED) + i for i in range(max(1, int(args.num_seeds)))]
    learned_like = [2, 1, 1]
    baseline = [0, 0, 0]

    base_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(base_logs_dir, exist_ok=True)

    pretrain_steps_list = [10000]

    def fmt_row(cols, widths):
        return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    for steps in pretrain_steps_list:
        setattr(Config, "INITIAL_TRAINING_STEPS", int(steps))
        print(f"\n=== Running baselines with INITIAL_TRAINING_STEPS={steps} ===")

        # 1) Action selection baseline comparison (MODE='random')
        setattr(Config, "MODE", "random")
        per_seed_random = []
        per_seed_random_success = []
        random_stats = {
            "cand_total": 0,
            "cand_eq_goal": 0,
            "sel_total": 0,
            "sel_eq_goal": 0,
            "imp_cand_blocked": 0,
            "imp_sel_blocked": 0,
        }
        actions_per_episode = len(learned_like)
        repeated_learned = learned_like * EPISODES_TO_AVG
        repeated_baseline = baseline * EPISODES_TO_AVG
        for s in seeds:
            run_result_learned = run_sequence(
                repeated_learned,
                seed=s,
                initial_steps=steps,
                return_metadata=True,
                return_success=True,
            )
            if isinstance(run_result_learned, tuple) and len(run_result_learned) == 3:
                r_learned, success_history_learned, metadata_random = run_result_learned  # type: ignore[misc]
            elif isinstance(run_result_learned, tuple) and len(run_result_learned) == 2:
                r_learned, metadata_random = run_result_learned  # type: ignore[misc]
                success_history_learned = []
            else:
                r_learned, metadata_random = run_result_learned, None
                success_history_learned = []

            if not success_history_learned and isinstance(metadata_random, dict):
                success_hist = [float(x) for x in metadata_random.get("success_history", [])]
                success_history_learned = success_hist

            if isinstance(metadata_random, dict):
                all_pairs = metadata_random.get("all_pairs", []) or []
                selected_pairs = metadata_random.get("selected_pairs", []) or []
                snapshot_maze = metadata_random.get("candidate_maze") or metadata_random.get("post_incremental_maze")
                random_stats["cand_total"] += len(all_pairs)
                random_stats["sel_total"] += len(selected_pairs)
                random_stats["cand_eq_goal"] += sum(
                    1
                    for p in all_pairs
                    if isinstance(p, (list, tuple)) and len(p) == 2 and int(p[0]) == int(p[1])
                )
                random_stats["sel_eq_goal"] += sum(
                    1
                    for p in selected_pairs
                    if isinstance(p, (list, tuple)) and len(p) == 2 and int(p[0]) == int(p[1])
                )
                if snapshot_maze is not None:
                    grid = int(len(snapshot_maze))

                    def goal_blocked(pair):
                        try:
                            g = int(pair[1])
                            r, c = divmod(g, grid)
                            return int(snapshot_maze[r][c]) == 1
                        except Exception:
                            return False

                    random_stats["imp_cand_blocked"] += sum(
                        1
                        for p in all_pairs
                        if isinstance(p, (list, tuple)) and len(p) == 2 and goal_blocked(p)
                    )
                    random_stats["imp_sel_blocked"] += sum(
                        1
                        for p in selected_pairs
                        if isinstance(p, (list, tuple)) and len(p) == 2 and goal_blocked(p)
                    )

                saved_path = save_valuation_dump(
                    base_logs_dir,
                    steps,
                    "random",
                    s,
                    metadata_random,
                    require_values=False,
                )
                if saved_path:
                    print(f"Saved valuation metadata for MODE=random, seed={s} -> {saved_path}")
                for stage_meta in metadata_random.get("stages", []):
                    stage_num = stage_meta.get("stage")
                    if stage_num is None:
                        continue
                    saved_stage = save_valuation_dump(
                        base_logs_dir,
                        steps,
                        "random",
                        s,
                        stage_meta,
                        stage=int(stage_num),
                        require_values=False,
                    )
                    if saved_stage:
                        print(f"  ↳ random stage {stage_num}: {saved_stage}")

            r_baseline, success_baseline = run_sequence(
                repeated_baseline,
                seed=s,
                initial_steps=steps,
                return_success=True,
            )
            learned_episode_rewards, learned_episode_success = _extract_episode_metrics(
                r_learned,
                success_history_learned,
                actions_per_episode,
            )
            baseline_episode_rewards, baseline_episode_success = _extract_episode_metrics(
                r_baseline,
                success_baseline,
                actions_per_episode,
            )

            learned_episode_rewards = learned_episode_rewards[:EPISODES_TO_AVG]
            baseline_episode_rewards = baseline_episode_rewards[:EPISODES_TO_AVG]
            learned_episode_success = learned_episode_success[:EPISODES_TO_AVG]
            baseline_episode_success = baseline_episode_success[:EPISODES_TO_AVG]

            avg_learned_reward = float(np.mean(learned_episode_rewards)) if learned_episode_rewards else 0.0
            avg_baseline_reward = float(np.mean(baseline_episode_rewards)) if baseline_episode_rewards else 0.0
            avg_learned_success = float(np.mean(learned_episode_success)) if learned_episode_success else 0.0
            avg_baseline_success = float(np.mean(baseline_episode_success)) if baseline_episode_success else 0.0

            per_seed_random.append((s, avg_learned_reward, avg_baseline_reward))
            per_seed_random_success.append((s, avg_learned_success, avg_baseline_success))
        learned_vals_random = [x[1] for x in per_seed_random]
        baseline_vals_random = [x[2] for x in per_seed_random]
        avg_learned_random = float(np.mean(learned_vals_random)) if learned_vals_random else 0.0
        avg_baseline_random = float(np.mean(baseline_vals_random)) if baseline_vals_random else 0.0
        sem_learned_random = float(np.std(learned_vals_random, ddof=1) / np.sqrt(len(learned_vals_random))) if len(learned_vals_random) > 1 else 0.0
        sem_baseline_random = float(np.std(baseline_vals_random, ddof=1) / np.sqrt(len(baseline_vals_random))) if len(baseline_vals_random) > 1 else 0.0

        learned_success_random = [x[1] for x in per_seed_random_success]
        baseline_success_random = [x[2] for x in per_seed_random_success]
        avg_learned_success_random = float(np.mean(learned_success_random)) if learned_success_random else 0.0
        avg_baseline_success_random = float(np.mean(baseline_success_random)) if baseline_success_random else 0.0
        sem_learned_success_random = (
            float(np.std(learned_success_random, ddof=1) / np.sqrt(len(learned_success_random)))
            if len(learned_success_random) > 1
            else 0.0
        )
        sem_baseline_success_random = (
            float(np.std(baseline_success_random, ddof=1) / np.sqrt(len(baseline_success_random)))
            if len(baseline_success_random) > 1
            else 0.0
        )

        print(
            f"Original data (MODE=random): {random_stats['cand_eq_goal']} / {random_stats['cand_total']} start-goal pairs "
            "had start=goal among candidates; "
            f"{random_stats['sel_eq_goal']} / {random_stats['sel_total']} selected."
        )
        print(
            f"Impossible goals (MODE=random): {random_stats['imp_cand_blocked']} / {random_stats['cand_total']} candidates "
            "had blocked goals; "
            f"{random_stats['imp_sel_blocked']} / {random_stats['sel_total']} selected."
        )

        # 2) Data valuation comparison for learned action selection only (optional)
        learned_modes = ["longest_paths", "top_with_mmr"] #top_with_mmr
        dv_results = {}
        if not args.skip_valuation:
            for mode in learned_modes:
                setattr(Config, "MODE", mode)
                per_seed = []
                per_seed_success = []
                # Aggregate counts across seeds
                cand_total = 0
                cand_eq_goal = 0
                sel_total = 0
                sel_eq_goal = 0
                imp_cand_blocked = 0
                imp_sel_blocked = 0
                for s in seeds:
                    run_result = run_sequence(
                        repeated_learned,
                        seed=s,
                        initial_steps=steps,
                        return_metadata=True,
                        return_success=True,
                    )
                    if isinstance(run_result, tuple) and len(run_result) == 3:
                        r_learned, success_history, metadata = run_result  # type: ignore[misc]
                    elif isinstance(run_result, tuple) and len(run_result) == 2:
                        r_learned, metadata = run_result  # type: ignore[misc]
                        success_history = []
                    else:
                        r_learned, metadata = run_result, None
                        success_history = []
                    episode_rewards, episode_success = _extract_episode_metrics(
                        r_learned,
                        success_history,
                        actions_per_episode,
                    )
                    if not success_history and isinstance(metadata, dict):
                        success_history = [float(x) for x in metadata.get("success_history", [])]
                        episode_rewards, episode_success = _extract_episode_metrics(
                            r_learned,
                            success_history,
                            actions_per_episode,
                        )
                    episode_rewards = episode_rewards[:EPISODES_TO_AVG]
                    episode_success = episode_success[:EPISODES_TO_AVG]
                    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                    avg_success = float(np.mean(episode_success)) if episode_success else 0.0
                    per_seed.append((s, avg_reward))
                    per_seed_success.append((s, avg_success))
                    # Tally start==goal and impossible-goal counts from metadata
                    if isinstance(metadata, dict):
                        all_pairs = metadata.get("all_pairs", []) or []
                        selected_pairs = metadata.get("selected_pairs", []) or []
                        snapshot_maze = metadata.get("candidate_maze") or metadata.get("post_incremental_maze")
                        cand_total += len(all_pairs)
                        sel_total += len(selected_pairs)
                        # start==goal
                        cand_eq_goal += sum(1 for p in all_pairs if isinstance(p, (list, tuple, list)) and len(p) == 2 and int(p[0]) == int(p[1]))
                        sel_eq_goal += sum(1 for p in selected_pairs if isinstance(p, (list, tuple, list)) and len(p) == 2 and int(p[0]) == int(p[1]))
                        # impossible-goal (blocked goal cell in current maze snapshot)
                        if snapshot_maze is not None:
                            grid = int(len(snapshot_maze))
                            def goal_blocked(pair):
                                try:
                                    g = int(pair[1])
                                    r, c = divmod(g, grid)
                                    return int(snapshot_maze[r][c]) == 1
                                except Exception:
                                    return False
                            imp_cand_blocked += sum(1 for p in all_pairs if isinstance(p, (list, tuple, list)) and len(p) == 2 and goal_blocked(p))
                            imp_sel_blocked += sum(1 for p in selected_pairs if isinstance(p, (list, tuple, list)) and len(p) == 2 and goal_blocked(p))
                    saved_path = save_valuation_dump(base_logs_dir, steps, mode, s, metadata)
                    if saved_path:
                        print(f"Saved valuation cache for MODE={mode}, seed={s} -> {saved_path}")
                    for stage_meta in metadata.get("stages", []):
                        stage_num = stage_meta.get("stage")
                        if stage_num is None:
                            continue
                        saved_stage = save_valuation_dump(
                            base_logs_dir,
                            steps,
                            mode,
                            s,
                            stage_meta,
                            stage=int(stage_num),
                            require_values=True,
                        )
                        if saved_stage:
                            print(f"  ↳ stage {stage_num}: {saved_stage}")
                vals = [x[1] for x in per_seed]
                avg_val = float(np.mean(vals)) if vals else 0.0
                sem_val = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                success_vals = [x[1] for x in per_seed_success]
                avg_success = float(np.mean(success_vals)) if success_vals else 0.0
                sem_success = (
                    float(np.std(success_vals, ddof=1) / np.sqrt(len(success_vals)))
                    if len(success_vals) > 1
                    else 0.0
                )
                dv_results[mode] = {
                    "per_seed": per_seed,
                    "avg": avg_val,
                    "sem": sem_val,
                    "per_seed_success": per_seed_success,
                    "avg_success": avg_success,
                    "sem_success": sem_success,
                    # extra stats
                    "cand_total": int(cand_total),
                    "cand_eq_goal": int(cand_eq_goal),
                    "sel_total": int(sel_total),
                    "sel_eq_goal": int(sel_eq_goal),
                    "imp_cand_blocked": int(imp_cand_blocked),
                    "imp_sel_blocked": int(imp_sel_blocked),
                }
                # Print summary lines
                print(
                    f"Original data (MODE={mode}): {cand_eq_goal} / {cand_total} start-goal pairs for which start=goal in candidates for which data value estimated. "
                    f"{sel_eq_goal} / {sel_total} start-goal pairs for which start=goal selected."
                )
                print(
                    f"Impossible goals (MODE={mode}): {imp_cand_blocked} / {cand_total} candidates had blocked goal cells. "
                    f"{imp_sel_blocked} / {sel_total} selected had blocked goal cells."
                )


        # Print to console for this pretrain setting
        print(f"Action selection baseline (MODE=random, averaged over {EPISODES_TO_AVG} meta-episodes):")
        for s, ml, mb in per_seed_random:
            print(f"  seed {s}: learned_mean={ml:.4f} baseline_mean={mb:.4f}")
        print(f"Averages: learned_mean={avg_learned_random:.4f} baseline_mean={avg_baseline_random:.4f}")

        print(f"\nAction selection success rate (MODE=random, averaged over {EPISODES_TO_AVG} meta-episodes):")
        for s, ml, mb in per_seed_random_success:
            print(f"  seed {s}: learned_success={ml:.4f} baseline_success={mb:.4f}")
        print(
            "Averages: learned_success="
            f"{avg_learned_success_random:.4f} baseline_success={avg_baseline_success_random:.4f}"
        )

        if not args.skip_valuation:
            print(f"\nData valuation (learned only, averaged over {EPISODES_TO_AVG} meta-episodes):")
            for mode in learned_modes:
                per_seed = dv_results[mode]["per_seed"]
                avg = dv_results[mode]["avg"]
                print(f"  MODE={mode}:")
                for s, ml in per_seed:
                    print(f"    seed {s}: learned_mean={ml:.4f}")
                print(f"    average: {avg:.4f}")
                per_seed_success = dv_results[mode]["per_seed_success"]
                avg_success = dv_results[mode]["avg_success"]
                print("    success rates:")
                for s, succ in per_seed_success:
                    print(f"      seed {s}: learned_success={succ:.4f}")
                print(f"      average_success: {avg_success:.4f}")

        # Write results for this pretrain setting
        out_dir = os.path.join(base_logs_dir, f"pretrain_{steps}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "results.txt")

        with open(out_path, "w") as f:
            # Table 1: action selection baseline
            f.write(
                f"Action selection baseline (MODE=random, averaged over {EPISODES_TO_AVG} meta-episodes)\n"
            )
            headers1 = ["seed", "learned_avg", "baseline_avg", "learned_sem", "baseline_sem"]
            rows1 = [[s, f"{ml:.4f}", f"{mb:.4f}", "", ""] for s, ml, mb in per_seed_random]
            rows1.append(["avg", f"{avg_learned_random:.4f}", f"{avg_baseline_random:.4f}", f"{sem_learned_random:.4f}", f"{sem_baseline_random:.4f}"])
            widths1 = [max(len(str(x[i])) for x in ([headers1] + rows1)) for i in range(len(headers1))]
            f.write(fmt_row(headers1, widths1) + "\n")
            f.write("-+-".join("-" * w for w in widths1) + "\n")
            for r in rows1:
                f.write(fmt_row(r, widths1) + "\n")

            # Table 1b: action selection success rate
            f.write(
                f"\nAction selection success rate (MODE=random, averaged over {EPISODES_TO_AVG} meta-episodes)\n"
            )
            headers1b = ["seed", "learned_success", "baseline_success", "learned_sem", "baseline_sem"]
            rows1b = [
                [s, f"{ml:.4f}", f"{mb:.4f}", "", ""]
                for s, ml, mb in per_seed_random_success
            ]
            rows1b.append(
                [
                    "avg",
                    f"{avg_learned_success_random:.4f}",
                    f"{avg_baseline_success_random:.4f}",
                    f"{sem_learned_success_random:.4f}",
                    f"{sem_baseline_success_random:.4f}",
                ]
            )
            widths1b = [max(len(str(x[i])) for x in ([headers1b] + rows1b)) for i in range(len(headers1b))]
            f.write(fmt_row(headers1b, widths1b) + "\n")
            f.write("-+-".join("-" * w for w in widths1b) + "\n")
            for r in rows1b:
                f.write(fmt_row(r, widths1b) + "\n")

            if not args.skip_valuation:
                f.write(
                    f"\nData valuation (learned only, averaged over {EPISODES_TO_AVG} meta-episodes)\n"
                )
                headers2 = ["mode", *[f"seed_{s}" for s in seeds], "avg", "sem"]
                widths2 = [max(len(h), 6) for h in headers2]
                f.write(fmt_row(headers2, widths2) + "\n")
                f.write("-+-".join("-" * w for w in widths2) + "\n")
                for mode in learned_modes:
                    per_seed = dv_results[mode]["per_seed"]
                    avg = dv_results[mode]["avg"]
                    sem = dv_results[mode]["sem"]
                    seed_vals = []
                    # Ensure order matches seeds list
                    s_to_v = {s: ml for s, ml in per_seed}
                    for s in seeds:
                        seed_vals.append(f"{s_to_v.get(s, 0.0):.4f}")
                    row = [mode, *seed_vals, f"{avg:.4f}", f"{sem:.4f}"]
                    # Update widths2 based on row content
                    widths2 = [max(w, len(str(c))) for w, c in zip(widths2, row)]
                    f.write(fmt_row(row, widths2) + "\n")

                # Start==Goal statistics
                f.write("\nStart==Goal statistics (aggregated across seeds)\n")
                headers_sg = ["mode", "candidates_eq/total", "selected_eq/total"]
                widths_sg = [max(len(h), 6) for h in headers_sg]
                f.write(fmt_row(headers_sg, widths_sg) + "\n")
                f.write("-+-".join("-" * w for w in widths_sg) + "\n")
                modes_for_stats = ["random", *learned_modes]
                for mode in modes_for_stats:
                    if mode == "random":
                        stats = random_stats
                    else:
                        stats = dv_results.get(mode, {})
                    cand_total = int(stats.get("cand_total", 0))
                    cand_eq = int(stats.get("cand_eq_goal", 0))
                    sel_total = int(stats.get("sel_total", 0))
                    sel_eq = int(stats.get("sel_eq_goal", 0))
                    row = [mode, f"{cand_eq} / {cand_total}", f"{sel_eq} / {sel_total}"]
                    widths_sg = [max(w, len(str(c))) for w, c in zip(widths_sg, row)]
                    f.write(fmt_row(row, widths_sg) + "\n")

                # Impossible-goal statistics
                f.write("\nImpossible-goal statistics (blocked goal cells; aggregated across seeds)\n")
                headers_imp = ["mode", "candidates_blocked/total", "selected_blocked/total"]
                widths_imp = [max(len(h), 6) for h in headers_imp]
                f.write(fmt_row(headers_imp, widths_imp) + "\n")
                f.write("-+-".join("-" * w for w in widths_imp) + "\n")
                for mode in modes_for_stats:
                    if mode == "random":
                        stats = random_stats
                    else:
                        stats = dv_results.get(mode, {})
                    cand_total = int(stats.get("cand_total", 0))
                    imp_cand = int(stats.get("imp_cand_blocked", 0))
                    sel_total = int(stats.get("sel_total", 0))
                    imp_sel = int(stats.get("imp_sel_blocked", 0))
                    row = [mode, f"{imp_cand} / {cand_total}", f"{imp_sel} / {sel_total}"]
                    widths_imp = [max(w, len(str(c))) for w, c in zip(widths_imp, row)]
                    f.write(fmt_row(row, widths_imp) + "\n")

                # Table 2b: success rates for data valuation
                f.write(
                    f"\nData valuation success rate (learned only, averaged over {EPISODES_TO_AVG} meta-episodes)\n"
                )
                headers2b = ["mode", *[f"seed_{s}" for s in seeds], "avg", "sem"]
                widths2b = [max(len(h), 6) for h in headers2b]
                f.write(fmt_row(headers2b, widths2b) + "\n")
                f.write("-+-".join("-" * w for w in widths2b) + "\n")
                for mode in learned_modes:
                    per_seed_success = dv_results[mode]["per_seed_success"]
                    avg_success = dv_results[mode]["avg_success"]
                    sem_success = dv_results[mode]["sem_success"]
                    s_to_v = {s: val for s, val in per_seed_success}
                    seed_vals = [f"{s_to_v.get(s, 0.0):.4f}" for s in seeds]
                    row = [mode, *seed_vals, f"{avg_success:.4f}", f"{sem_success:.4f}"]
                    widths2b = [max(w, len(str(c))) for w, c in zip(widths2b, row)]
                    f.write(fmt_row(row, widths2b) + "\n")

        print(f"\nSaved results to: {out_path}")
if __name__ == "__main__":
    main()
