## Image task

This README describes what is executed inside the `Main experiment.ipynb` notebook in this folder.

The notebook consists of the following cells, run in order:

1. `%run experiment.py`  
   Runs the main image experiment, constructing the image meta‑environment (wrapping Fashion‑MNIST with noisy corruptions and replay/generator actions) and training a recurrent PPO meta‑controller on top of it. The script covers both the non‑curriculum and curriculum learning regimes, logging episode rewards, action probabilities, and other diagnostics needed for later analysis. Models, logs, and primary training curves are written into the `logs/` directory (and any default `plots/` output used in the notebook) during this run.

2. `%run baselines_comparison.py`  
   Evaluates a set of non‑learning or simple heuristic baselines, and optionally a pre‑trained meta‑controller, under the same image meta‑environment settings. It reuses the configuration implied by `config.py` and the notebook, then records reward statistics and action‑usage patterns for each baseline. The script writes comparison metrics and figures into `logs/` and `plots/` for direct comparison with the learned controller.

3. `%run data_valuation_comparison.py --episodes 10`  
   Runs a targeted comparison of different data‑valuation approaches (for example, Shapley‑based value estimates versus simpler or random proxies) inside the same image meta‑environment. Over 10 evaluation episodes, it varies the valuation method and replay mechanism, logging how these choices affect downstream task performance and sample usage. The resulting summary tables and figures, written into `plots/` and `logs/`, support the comparison of which value signals are most effective.

4. `%run data_value_vs_novelty.py --novelty-metric distance`  
   Measures how the learned or assumed data value correlates with a particular notion of novelty, here given by a distance‑based metric in feature space. It runs short episodes or batched evaluations in the image meta‑environment, computing both the predicted value and the chosen novelty score for many examples, then aggregates these into correlation statistics (e.g., Pearson \(r\)). The script saves the resulting plots and numeric summaries into `plots/`, showing whether high‑valued data points are also more novel under this metric.

