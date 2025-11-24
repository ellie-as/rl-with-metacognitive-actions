## Maze task

This README describes what is executed inside the `Main experiment.ipynb` notebook in this folder.

The notebook consists of the following cells, run in order:

1. `%run experiment.py`  
   Runs the main maze experiment, training a PPO meta‑controller that chooses between different replay/world‑model actions in the maze meta‑environment. The script sets up the environment, runs training for a fixed number of timesteps, periodically evaluates the controller, and records summary statistics as well as per‑step action probabilities. Logs, checkpoints, and training curves are written into the `logs/` and `plots/` subdirectories.

2. `%run plot_training_actions.py`  
   Reads the action‑probability logs produced by `experiment.py` (for example `logs/action_probs_train.csv`) and aggregates them into a time‑series view of how often each meta‑action is chosen. The script computes rolling averages over training steps, optionally smooths the curves, and renders the figure `actions over time.png` into the current folder as a summary of the learned policy’s behaviour over training.

3. `%run action_selection_baselines_incremental.py --num-seeds 3`  
   Runs a set of incremental, hand‑designed action‑selection baselines that do not learn but instead follow fixed or simple policies over meta‑actions. For each of 3 random seeds, the script reuses the same underlying maze setup as the main experiment and logs the resulting rewards and action traces for these baseline strategies. The outputs (including any summary text files and auxiliary plots) are written under `logs/` for comparison with the learned PPO controller.

4. `!python stage_path_change_analysis.py ...`  
   Performs a post‑hoc analysis of cached valuation results saved earlier (for the `top_with_mmr` mode across the listed stages). It aggregates how path lengths and path values change over stages and across replay configurations, computing summary statistics that highlight any systematic biases. The script then generates a set of figures and writes them into the `plots/` directory.

