import pandas as pd
import matplotlib.pyplot as plt

csv_path   = "logs/action_probs_train.csv"   # change if needed
roll_window = 100                           # rolling‑mean window

df = pd.read_csv(csv_path)
action_cols = ["prob_action_0", "prob_action_1", "prob_action_2"]
step_ids    = sorted(df["step_in_episode"].unique())

action_labels = ["Memory replay", "Generative replay", "Update WM"]
action_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # blue=MR, green=GR, orange=Update WM

# ── create a 1×N grid of sub‑plots ──────────────────────────────────
n_steps = len(step_ids)
fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4), sharey=True)

for ax, step in zip(axes, step_ids):
    sub = df[df["step_in_episode"] == step].reset_index(drop=True)

    mean = sub[action_cols].rolling(roll_window, min_periods=1).mean()
    std  = sub[action_cols].rolling(roll_window, min_periods=1).std().fillna(0)

    for idx, col in enumerate(action_cols):
        ax.plot(mean.index, mean[col], label=action_labels[idx], color=action_colors[idx])
        ax.fill_between(mean.index,
                        mean[col] - std[col],
                        mean[col] + std[col],
                        alpha=0.2,
                        color=action_colors[idx])

    ax.set_title(f"Step {step} in episode")
    ax.set_xlabel("Training episode")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Probability")
axes[0].legend(loc="upper right", bbox_to_anchor=(0.5, -0.3), ncols=3)  # single legend
fig.suptitle("Rolling action probabilities (window = "
             f"{roll_window})", y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig('actions over time.png', dpi=200, bbox_inches='tight')