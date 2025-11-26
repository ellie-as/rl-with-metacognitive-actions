import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from visualisation import inference_bar_plot_across_runs
from family_tree_task import FamilyTreeBuilder
from spatial_task import SpatialBuilder

labels= ["Learn from observations", 
         "Consolidate observations into a graph",
         "Learn from inferred graph",
         "Expand graph with world model",
         "Update world model",
         "Do nothing"]

model_dirs     = [
    "SpatialBuilder_results",
    "FamilyTreeBuilder_results"
]

n_runs         = 1
step_positions = list(range(5))
action_cols    = [f"action_{i}_prob" for i in range(6)]
rolling_window = 1000 

for model_dir in model_dirs:
    # load and align rewards
    reward_runs = []
    for run in range(1, n_runs+1):
        r = pd.read_csv(os.path.join(model_dir, f"run_{run}_rewards.csv"))
        reward_runs.append(r["episode_reward"].values)
    min_eps     = min(len(r) for r in reward_runs)
    reward_arr  = np.stack([r[:min_eps] for r in reward_runs], axis=0)
    episodes    = np.arange(min_eps)

    if n_runs == 1:
        # Single-run: use rolling mean ± rolling std over episodes (matches image/maze style)
        series_mean = pd.Series(reward_arr[0])
        reward_mean_smooth = (
            series_mean.rolling(window=rolling_window, center=True, min_periods=1)
            .mean()
            .values
        )
        reward_std = (
            series_mean.rolling(window=rolling_window, center=True, min_periods=1)
            .std(ddof=0)
            .fillna(0.0)
            .values
        )
        sem_reward_smooth = reward_std
    else:
        # Multi-run: across-run mean ± SEM, both smoothed over episodes
        mean_reward = reward_arr.mean(axis=0)
        sem_reward  = reward_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)

        reward_mean_smooth = pd.Series(mean_reward)\
                                .rolling(window=rolling_window,
                                         center=True,
                                         min_periods=1)\
                                .mean().values
        sem_reward_smooth  = pd.Series(sem_reward)\
                                .rolling(window=rolling_window,
                                         center=True,
                                         min_periods=1)\
                                .mean().values

    # load and align action probs
    action_data = {p: {} for p in step_positions}
    for run in range(1, n_runs+1):
        df = pd.read_csv(os.path.join(model_dir, f"run_{run}_action_probs.csv"))
        for p in step_positions:
            df_p = (
                df[df["step_in_episode"] == p]
                  .set_index("episode_index")
                  .sort_index()
            )
            action_data[p][run] = df_p[action_cols]

    mean_smooth = {}
    sem_smooth  = {}
    common_eps  = {}

    for p in step_positions:
        # intersect episode indices across runs
        idx_sets = [set(df.index) for df in action_data[p].values()]
        eps = sorted(set.intersection(*idx_sets))
        common_eps[p] = eps

        arr = np.stack([
            action_data[p][run].loc[eps].values
            for run in range(1, n_runs+1)
        ], axis=0)  # shape: (runs, len(eps), 6)

        if n_runs == 1:
            # Single-run: rolling mean ± rolling std over episodes for each action
            raw = arr[0]  # (len(eps), 6)
            mean_mat = []
            std_mat = []
            for a in range(raw.shape[1]):
                s = pd.Series(raw[:, a])
                m = (
                    s.rolling(window=rolling_window, center=True, min_periods=1)
                    .mean()
                    .values
                )
                sd = (
                    s.rolling(window=rolling_window, center=True, min_periods=1)
                    .std(ddof=0)
                    .fillna(0.0)
                    .values
                )
                mean_mat.append(m)
                std_mat.append(sd)
            mean_smooth[p] = np.stack(mean_mat, axis=1)
            sem_smooth[p] = np.stack(std_mat, axis=1)
        else:
            # Multi-run: across-run mean ± SEM, both smoothed
            raw_mean = arr.mean(axis=0)  # (len(eps), 6)
            raw_sem  = arr.std(axis=0, ddof=1) / np.sqrt(n_runs)

            mean_smooth[p] = np.stack([
                pd.Series(raw_mean[:, a])
                  .rolling(window=rolling_window,
                           center=True,
                           min_periods=1)
                  .mean().values
                for a in range(raw_mean.shape[1])
            ], axis=1)

            sem_smooth[p] = np.stack([
                pd.Series(raw_sem[:, a])
                  .rolling(window=rolling_window,
                           center=True,
                           min_periods=1)
                  .mean().values
                for a in range(raw_sem.shape[1])
            ], axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f"{model_dir}  (smoothed ± window={rolling_window})", fontsize=16)
    ax_flat = axes.flatten()

    # Reward subplot
    ax_flat[0].plot(episodes, reward_mean_smooth, label="mean reward (smoothed)")
    ax_flat[0].fill_between(
        episodes,
        reward_mean_smooth - sem_reward_smooth,
        reward_mean_smooth + sem_reward_smooth,
        alpha=0.3
    )
    ax_flat[0].set_title("Mean episode reward")
    ax_flat[0].set_xlabel("Training episode")
    ax_flat[0].set_ylabel("Reward")

    # Action-prob subplots
    for i, p in enumerate(step_positions, start=1):
        eps = common_eps[p]
        for a in range(6):
            m = mean_smooth[p][:, a]
            s = sem_smooth[p][:, a]
            ax_flat[i].plot(eps, m, label=labels[a])
            ax_flat[i].fill_between(eps, m - s, m + s, alpha=0.2)
        ax_flat[i].set_title(f"Step {p} in Episode")
        ax_flat[i].set_xlabel("Training episode")
        ax_flat[i].set_ylabel("Probability")

    handles, legend_labels = ax_flat[1].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{model_dir}.png', dpi=200, bbox_inches='tight')
    plt.show()

    # === Inference bar chart from trained models ===
    # Determine builder from directory name
    if model_dir.startswith("FamilyTreeBuilder"):
        builder = FamilyTreeBuilder(base_num_children=2, grandparent_num_children=2)
    elif model_dir.startswith("SpatialBuilder"):
        builder = SpatialBuilder()
    else:
        print(f"[WARN] Unknown builder for results directory: {model_dir}. Skipping inference bar plot.")
        continue

    # Load trained models if present
    models = []
    for run in range(1, n_runs + 1):
        model_path = os.path.join(model_dir, f"run_{run}_model.zip")
        if os.path.exists(model_path):
            try:
                models.append(RecurrentPPO.load(model_path))
            except Exception as e:
                print(f"[WARN] Failed to load model '{model_path}': {e}")
        else:
            print(f"[WARN] Missing trained model file: {model_path}")

    if len(models) == 0:
        print(f"[WARN] No trained models found in '{model_dir}'. Skipping inference bar plot.")
        continue

    # Use the same number of meta-steps as plotted (caps to first 5 steps)
    max_meta_steps = max(step_positions) + 1 if len(step_positions) > 0 else 5

    # This will save '{builder_name}_results/inference_bar_across_runs.png'
    inference_bar_plot_across_runs(
        builder=builder,
        models=models,
        n_eval_episodes=200,
        max_meta_steps=max_meta_steps,
        seed_value=None,
    )
