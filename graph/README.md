## Graph task

This README describes what is executed inside the `Main experiment.ipynb` notebook in this folder.

The notebook consists of the following cells, run in order:

1. `!python metacontroller.py`  
   Runs the main graph experiment by invoking the `metacontroller.py` script with its internal default arguments. The script constructs graph‑based meta‑environments for both the family‑tree and spatial‑relation tasks, then trains a recurrent PPO meta‑controller that decides which high‑level operation (e.g. training from the store, expanding the graph with a GCN) to apply at each meta‑step. During training it logs rewards, action probabilities, and other diagnostics, and saves trained controller checkpoints and any automatically generated figures into the appropriate `logs/` and `plots/` subdirectories.

2. `!python recreate_plots.py`  
   Re‑runs the plotting pipeline without retraining any models. The `recreate_plots.py` script reloads the logs and checkpoints produced by `metacontroller.py`, recomputes any aggregated statistics across runs, and regenerates the main figures (such as performance curves and action‑usage summaries). All regenerated plots are written into the `plots/` directory.

