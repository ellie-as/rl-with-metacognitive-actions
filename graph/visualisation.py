import os
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import torch
import torch as th
import gymnasium as gym

from environment import GraphEnv
from seed_utils import derive_seed, seed_all


def rolling_mean(data, window=50):
    """
    1D convolve-based rolling mean.
    If data length < window, returns data unchanged.
    """
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def rolling_arr_simple(x, w=50):
    """
    More robust rolling approach for partial data:
      1) Extract non-NaN data contiguously
      2) Apply rolling_mean(...)
      3) Place the rolled result at the start
    """
    valid = ~np.isnan(x)
    valid_data = x[valid]
    if len(valid_data) < w:
        return x
    rolled_valid = rolling_mean(valid_data, w)
    out = np.full_like(x, np.nan)
    out[:len(rolled_valid)] = rolled_valid
    return out


def rolling_mean_std_1d(x, window=50):
    """
    Compute rolling mean and std for a 1D array using a non-centered window.
    NaNs are handled by pandas' rolling; positions where the entire window
    is NaN get std=0 so the band collapses to the mean.
    """
    s = pd.Series(x)
    mean = s.rolling(window=window, center=False, min_periods=1).mean()
    std = s.rolling(window=window, center=False, min_periods=1).std(ddof=0)
    std = std.fillna(0.0)
    return mean.to_numpy(), std.to_numpy()

def plot_aggregated_rewards(steps_list, rewards_list, window=50, output_path="aggregated_rewards.png"):
    if not rewards_list:
        print("[WARN] No episodes to plot for rewards.")
        return

    n_runs = len(rewards_list)

    # Single-run: use rolling mean ± rolling std over episodes (matches maze/image).
    if n_runs == 1:
        rewards = rewards_list[0]
        max_episodes = len(rewards)
        if max_episodes == 0:
            print("[WARN] No episodes to plot for rewards.")
            return
        mean_smoothed, std_smoothed = rolling_mean_std_1d(rewards, window)
        x_vals = np.arange(max_episodes)
    else:
        max_episodes=max(len(r) for r in rewards_list)
        if max_episodes==0:
            print("[WARN] No episodes to plot for rewards.")
            return

        all_rewards_padded=[]
        for r in rewards_list:
            padded=np.full((max_episodes,), np.nan)
            padded[:len(r)]=r
            all_rewards_padded.append(padded)
        all_rewards_padded=np.array(all_rewards_padded)

        mean_rewards=np.nanmean(all_rewards_padded, axis=0)
        std_rewards=np.nanstd(all_rewards_padded, axis=0)

        mean_smoothed=rolling_arr_simple(mean_rewards, window)
        std_smoothed=rolling_arr_simple(std_rewards, window)

        x_vals=np.arange(max_episodes)
    lower=mean_smoothed - std_smoothed
    upper=mean_smoothed + std_smoothed

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, mean_smoothed, label="Mean Reward", color="blue")
    plt.fill_between(x_vals, lower, upper, alpha=0.3, color="blue", label="±1 std")
    plt.xlabel("Episode index")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved aggregated reward plot to {output_path}")


def plot_aggregated_action_probs(action_df, n_actions, max_episode_steps, window=50, output_dir="results"):
    """
    Plot training-time action probabilities.
    We unify color array with inference code below, so actions match color consistently.
    """
    labels_dict={
        0:"Learn from memories",
        1:"Transfer memories into graph",
        2:"Learn from graph",
        3:"Expand graph",
        4:"Update rule library",
        5:"Do nothing"
    }
    colors=["blue","green","red","orange","purple","brown"]

    os.makedirs(output_dir, exist_ok=True)

    grouped=action_df.groupby("step_in_episode")
    for step_i in range(max_episode_steps):
        if step_i not in grouped.groups:
            continue
        step_df=grouped.get_group(step_i)
        run_groups=step_df.groupby("run_id")
        n_runs=len(run_groups)
        max_ep=step_df["episode_index"].max()+1

        # Single-run: rolling mean ± rolling std over episodes for this step.
        if n_runs == 1:
            rdata_sorted = step_df.sort_values("episode_index")
            prob_array = np.full((max_ep, n_actions), np.nan, dtype=np.float32)
            for row in rdata_sorted.itertuples():
                ep_idx=getattr(row,"episode_index")
                for a in range(n_actions):
                    col_name=f"action_{a}_prob"
                    prob_val=getattr(row,col_name)
                    prob_array[ep_idx, a]=prob_val

            x_vals = np.arange(max_ep)
            plt.figure(figsize=(6,4))
            for a in range(n_actions):
                label=labels_dict[a]
                color=colors[a%len(colors)]
                y_raw = prob_array[:, a]
                mean_y, std_y = rolling_mean_std_1d(y_raw, window)
                lower = mean_y - std_y
                upper = mean_y + std_y
                plt.plot(x_vals, mean_y, label=label, color=color)
                plt.fill_between(x_vals, lower, upper, alpha=0.2, color=color)
        else:
            prob_array=np.full((n_runs, max_ep, n_actions), np.nan, dtype=np.float32)

            for idx, (rid, rdata) in enumerate(run_groups):
                rdata_sorted=rdata.sort_values("episode_index")
                for row in rdata_sorted.itertuples():
                    ep_idx=getattr(row,"episode_index")
                    for a in range(n_actions):
                        col_name=f"action_{a}_prob"
                        prob_val=getattr(row,col_name)
                        prob_array[idx, ep_idx, a]=prob_val

            mean_probs=np.nanmean(prob_array, axis=0)
            std_probs=np.nanstd(prob_array, axis=0)

            for a in range(n_actions):
                mean_probs[:, a]=rolling_arr_simple(mean_probs[:, a], window)
                std_probs[:, a]=rolling_arr_simple(std_probs[:, a], window)

            x_vals=np.arange(max_ep)
            plt.figure(figsize=(6,4))
            for a in range(n_actions):
                label=labels_dict[a]
                color=colors[a%len(colors)]
                y=mean_probs[:, a]
                y_std=std_probs[:, a]
                lower=y-y_std
                upper=y+y_std
                plt.plot(x_vals, y, label=label, color=color)
                plt.fill_between(x_vals, lower, upper, alpha=0.2, color=color)

        plt.title(f"Mean action probabilities at step {step_i}")
        plt.xlabel("Episode index")
        plt.ylabel("Probability")
        if step_i==0:
            plt.legend()
        plt.tight_layout()
        save_path=os.path.join(output_dir, f"action_probs_step_{step_i}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved aggregated action prob plot for step {step_i} to {save_path}")


def plot_combined_2row4col(builder_name, steps_list, rewards_list, action_df,
                           n_actions=6, max_episode_steps=10, window=50):
    """
    Produce a single figure with 2x4 subplots:
     (0,0) = aggregated reward
     (0,1) = step 0
     (0,2) = step 1
     (0,3) = step 2
     (1,0) = step 3
     (1,1) = step 4
     (1,2) = step 5
     (1,3) = step 6
    """
    results_dir=f"{builder_name}_results"
    os.makedirs(results_dir, exist_ok=True)

    fig, axes=plt.subplots(nrows=2, ncols=4, figsize=(20,10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # top-left subplot => aggregated reward
    ax_reward=axes[0,0]
    max_episodes=max(len(r) for r in rewards_list) if rewards_list else 0
    if max_episodes==0:
        ax_reward.set_title("No Episodes (Empty)")
    else:
        n_runs = len(rewards_list)
        if n_runs == 1:
            rewards = rewards_list[0]
            mean_smoothed, std_smoothed = rolling_mean_std_1d(rewards, window)
            x_vals = np.arange(len(rewards))
        else:
            # pad & compute across runs
            all_rewards_padded=[]
            for r in rewards_list:
                padded=np.full((max_episodes,), np.nan)
                padded[:len(r)]=r
                all_rewards_padded.append(padded)
            all_rewards_padded=np.array(all_rewards_padded)
            mean_rewards=np.nanmean(all_rewards_padded, axis=0)
            std_rewards=np.nanstd(all_rewards_padded, axis=0)

            mean_smoothed=rolling_arr_simple(mean_rewards, window)
            std_smoothed=rolling_arr_simple(std_rewards, window)

            x_vals=np.arange(max_episodes)

        lower=mean_smoothed-std_smoothed
        upper=mean_smoothed+std_smoothed
        ax_reward.plot(x_vals, mean_smoothed, label="Mean Reward", color="blue")
        ax_reward.fill_between(x_vals, lower, upper, alpha=0.3, color="blue")
        ax_reward.set_xlabel("Episode index")
        ax_reward.set_ylabel("Reward")
        ax_reward.set_title("Aggregated Reward")
        ax_reward.legend()

    labels_dict={
        0:"Learn from memories",
        1:"Transfer memories",
        2:"Learn from graph",
        3:"Expand graph",
        4:"Update rule library",
        5:"Do nothing"
    }
    colors=["blue","green","red","orange","purple","brown"]

    grouped=action_df.groupby("step_in_episode")

    def plot_step_in_axis(step_i, ax):
        if step_i not in grouped.groups:
            ax.set_title(f"No data for step {step_i}")
            return
        step_df=grouped.get_group(step_i)
        run_groups=step_df.groupby("run_id")
        max_ep=step_df["episode_index"].max()+1
        n_runs=len(run_groups)

        # Single-run: rolling mean ± rolling std over episodes for this step.
        if n_runs == 1:
            rdata_sorted = step_df.sort_values("episode_index")
            prob_array=np.full((max_ep, n_actions), np.nan, dtype=np.float32)
            for row in rdata_sorted.itertuples():
                ep_idx=getattr(row,"episode_index")
                for a in range(n_actions):
                    col_name=f"action_{a}_prob"
                    prob_val=getattr(row,col_name)
                    prob_array[ep_idx, a]=prob_val

            x_vals=np.arange(max_ep)
            for a in range(n_actions):
                label=labels_dict[a]
                color=colors[a%len(colors)]
                y_raw=prob_array[:, a]
                mean_y, std_y = rolling_mean_std_1d(y_raw, window)
                lower=mean_y-std_y
                upper=mean_y+std_y
                ax.plot(x_vals, mean_y, label=label, color=color)
                ax.fill_between(x_vals, lower, upper, alpha=0.2, color=color)
        else:
            prob_array=np.full((n_runs, max_ep, n_actions), np.nan, dtype=np.float32)

            for idx,(rid,rdata) in enumerate(run_groups):
                rdata_sorted=rdata.sort_values("episode_index")
                for row in rdata_sorted.itertuples():
                    ep_idx=getattr(row,"episode_index")
                    for a in range(n_actions):
                        col_name=f"action_{a}_prob"
                        prob_val=getattr(row,col_name)
                        prob_array[idx, ep_idx, a]=prob_val
            mean_probs=np.nanmean(prob_array, axis=0)
            std_probs=np.nanstd(prob_array, axis=0)
            for a in range(n_actions):
                mean_probs[:, a]=rolling_arr_simple(mean_probs[:, a], window)
                std_probs[:, a]=rolling_arr_simple(std_probs[:, a], window)

            x_vals=np.arange(max_ep)
            for a in range(n_actions):
                label=labels_dict[a]
                color=colors[a%len(colors)]
                y=mean_probs[:, a]
                y_std=std_probs[:, a]
                lower=y-y_std
                upper=y+y_std
                ax.plot(x_vals, y, label=label, color=color)
                ax.fill_between(x_vals, lower, upper, alpha=0.2, color=color)
        ax.set_xlabel("Episode index")
        ax.set_ylabel("Probability")
        ax.set_title(f"Step {step_i}")

    step_positions=[
        (0,1,0),
        (0,2,1),
        (0,3,2),
        (1,0,3),
        (1,1,4),
        (1,2,5),
        (1,3,6),
    ]
    for (row,col,step_i) in step_positions:
        plot_step_in_axis(step_i, axes[row][col])

    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_actions)
    save_path=os.path.join(results_dir, "combined_plots.png")
    plt.tight_layout(rect=[0,0.07,1,1])
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved single combined figure (2x4 subplots) to {save_path}")


from typing import Optional

def inference_bar_plot_across_runs(builder, models, n_eval_episodes=20, max_meta_steps=10, seed_value: Optional[int] = None):
    """
    1) For each model in `models`, run n_eval_episodes.
    2) Gather action frequencies + observation factors
    3) Average freq across models, produce grouped bar chart
    4) Also produce a line chart of observation factors (store_count,graph_edge_count,rule_flag).
    """

    n_actions=6
    labels_dict={
        0:"Learn from observations",
        1:"Consolidate observations into a graph",
        2:"Learn from inferred graph",
        3:"Expand graph with world model",
        4:"Update world model",
        5:"Do nothing"
    }

    colors=["C0","orange","green","red","purple","brown"]

    builder_name=builder.__class__.__name__
    results_dir=f"{builder_name}_results"
    os.makedirs(results_dir, exist_ok=True)

    freq_models = np.zeros((len(models), max_meta_steps, n_actions), dtype=np.float32)
    freq_models_std = np.zeros((len(models), max_meta_steps, n_actions), dtype=np.float32)
    all_obs = []

    resolved_seed = seed_value
    if resolved_seed is not None:
        seed_all(resolved_seed)

    for m_idx, model_or_path in enumerate(models):
        # Load model if a path string was provided
        if isinstance(model_or_path, str):
            model = RecurrentPPO.load(model_or_path, device="cpu")
        else:
            model = model_or_path
        model_seed = derive_seed(resolved_seed or 0, "inference_model", m_idx) if resolved_seed is not None else None
        if model_seed is not None:
            seed_all(model_seed)
        # create test env
        test_env = GraphEnv(builder, results_dir, max_meta_steps, initial_obs_count=20)
        if model_seed is not None:
            test_env.action_space.seed(model_seed)
        actions_record = np.full((n_eval_episodes, max_meta_steps), -1, dtype=int)
        obs_record = np.zeros((n_eval_episodes, max_meta_steps, 3), dtype=np.float32)

        for ep_idx in range(n_eval_episodes):
            if model_seed is not None:
                episode_seed = derive_seed(model_seed, "inference_episode", ep_idx)
                seed_all(episode_seed)
                obs, _ = test_env.reset(seed=episode_seed)
            else:
                obs,_=test_env.reset()
            done=False
            step_i=0
            lstm_states=None
            while not done and step_i<max_meta_steps:
                obs_record[ep_idx, step_i, :]=obs[:3]
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                actions_record[ep_idx, step_i]=action
                obs, _, done, _, _=test_env.step(action)
                step_i+=1

        freq_of_action=np.zeros((max_meta_steps,n_actions), dtype=np.float32)
        freq_of_action_std=np.zeros((max_meta_steps,n_actions), dtype=np.float32)
        for step_i in range(max_meta_steps):
            step_actions=actions_record[:,step_i]
            valid_mask=(step_actions!=-1)
            valid_actions=step_actions[valid_mask]
            n_valid=len(valid_actions)
            if n_valid==0:
                continue
            for a in range(n_actions):
                count_a = np.sum(valid_actions==a)
                frac_a = count_a/n_valid
                freq_of_action[step_i, a] = frac_a
                freq_of_action_std[step_i, a] = np.sqrt(frac_a*(1.0-frac_a)/n_valid)

        freq_models[m_idx,:,:]=freq_of_action
        freq_models_std[m_idx,:,:]=freq_of_action_std
        all_obs.append(obs_record)

        # Ensure test env and model references are released promptly
        try:
            test_env.close()
        except Exception:
            pass
        del test_env
        if isinstance(model_or_path, str):
            del model

    mean_freq=np.mean(freq_models, axis=0)
    mean_std=np.mean(freq_models_std, axis=0)  # naive average of std

    # # group bar chart
    # x=np.arange(max_meta_steps)
    # bar_width=0.24
    # fig, ax=plt.subplots(figsize=(10,4))
    # for a in range(n_actions):
    #     offsets = x + (a-(n_actions/2 -0.5))*bar_width
    #     ax.bar(offsets,
    #            mean_freq[:,a],
    #            bar_width,
    #            yerr=mean_std[:,a],
    #            label=labels_dict[a],
    #            color=colors[a%len(colors)],
    #            capsize=3)

    # ax.set_xlabel("Step in episode")
    # ax.set_ylabel("Action frequency")
    # ax.set_xticks(x)
    # ax.set_xticklabels([str(i) for i in range(max_meta_steps)])

    # number of steps to plot
    n_plot = 5  
    # build x‐axis [0,1,2,3,4]
    x = np.arange(n_plot)
    bar_width=0.14
    
    fig, ax = plt.subplots(figsize=(8,2))
    for a in range(n_actions):
        offsets = x + (a - (n_actions/2 - 0.5)) * bar_width
        ax.bar(offsets,
               mean_freq[:n_plot, a],        # only first 5 steps
               bar_width,
               yerr=mean_std[:n_plot, a],    # only first 5 steps
               label=labels_dict[a],
               color=colors[a%len(colors)],
               capsize=3,
               alpha=0.5)
    
    ax.set_xlabel("Step in episode")
    ax.set_ylabel("Action frequency")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_plot)])
    plt.tight_layout()
    
    #ax.legend(loc="best")
    plt.tight_layout()
    
    out_bar=os.path.join(results_dir, "inference_bar_across_runs.png")
    plt.savefig(out_bar, dpi=200)
    plt.close()
    print(f"[INFO] Saved averaged inference bar chart for {builder_name} to {out_bar}")

    # obs factor
    merged_obs = np.concatenate(all_obs, axis=0)  # shape=(num_models*n_eval_episodes, max_meta_steps, 3)
    mean_obs = np.mean(merged_obs, axis=0)
    std_obs  = np.std(merged_obs, axis=0)

    factor_names=["Store Count","Graph Edge Count","Rule Flag"]
    fig2, axarr=plt.subplots(nrows=1, ncols=3, figsize=(12,3))
    steps=np.arange(max_meta_steps)
    for i in range(3):
        axarr[i].errorbar(steps, mean_obs[:,i], yerr=std_obs[:,i],
                          fmt='-o', color='blue', ecolor='lightblue', capsize=3)
        axarr[i].set_title(factor_names[i])
        axarr[i].set_xlabel("Step")
        axarr[i].set_ylabel("Mean Value")
    fig2.suptitle("Observation Factors Over Meta-Steps (Inference)")

    plt.tight_layout()
    out_obs=os.path.join(results_dir,"obs_factors_inference.png")
    plt.savefig(out_obs, dpi=200)
    plt.close()
    print(f"[INFO] Saved observation factors plot to {out_obs}")
