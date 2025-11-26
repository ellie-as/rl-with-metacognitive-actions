## Modelling the control of offline processing with reinforcement learning

*Code for the accepted NeurIPS 2025 paper 'Modelling the control of offline processing with reinforcement learning'.*

Brains use different kinds of 'metacognitive actions' to reorganise their representations in offline learning, such as replay of hippocampal memories, consolidation of experience into a world model, and generative replay based on this world model. The result of these different processes depends on lots of other variables, like how well the world model captures the environment, and the novelty of the environment. One might hypothesise that there is an ideal 'curriculum' for offline learning, but this can be hard to hand-engineer. Instead, can a meta-controller use these 'metacognitive actions' to learn offline in the most effective way? Furthermore can the system select which experiences are most useful to replay or simulate? 

The simulations involve a 'meta-controller' RL agent that co-ordinates offline learning of a task-specific model / agent. In the maze and image cases, it has three actions to choose between:
* Update the task-specific model / agent based on real memories from the model hippocampus
* Update the 'world model' (in the maze case this captures transition statistics, whilst in the image case it is a generative model of images)
* Update the task-specific model / agent based on samples from the world model

![Main diagram of the meta‑controller and offline learning setup](diagrams/main%20diagram.png)

In the relational knowledge case there are additional actions to illustrate how more complicated abstractions can be learned:

![Graph task diagram](diagrams/graph%20diagram.png)

### Meta‑controller training

Across all three tasks, the meta‑controller is a recurrent PPO agent (`RecurrentPPO` with an LSTM policy from `sb3_contrib`) with a discrete action space that chooses which offline operation to apply next. The agent is trained on short meta‑episodes: in the image and maze tasks, intermediate steps have zero reward and the final step receives a scalar reward summarising the performance of the lower-level model / agent in the `wake state' (test accuracy in the image task; mean validation reward in the maze). In the graph task, each non‑trivial meta‑action incurs a small cost and the episode terminates with a reward equal to the number of correctly learned relations.

The meta‑controller observes low‑dimensional state summaries rather than raw inputs. In the **image task**, the meta-controller's observation consists of (i) the current meta-step count, (ii) per‑class fractions of items currently in the replay buffer (i.e. the empirical class distribution in the hippocampus), and (iii) the previous meta‑action. In the **maze task**, the observation consists of (i) the current meta‑step count, (ii) the world‑model accuracy, and (iii) the previous meta‑action. In the **graph task**, the observation consists of (i) the number of facts stored in the MF learner, (ii) the number of edges in the current working graph, and (iii) a binary flag indicating whether the GCN‑based rule library has been trained.

### Reproducing the results

Simulations can be run with:

```commandline
pip install -r requirements.txt
```

... then run the relevant notebook.

This code has been tested using **Python 3.9.6**, on Linux virtual machines with NVIDIA A100 GPUs, and on MacOS with the MPS backend for GPU support.

To reproduce the paper's figures:

#### Figure 2 (image task):
* Open `./image/Main experiment.ipynb` and run all cells.  
  This:
  - trains the meta‑controller on Fashion‑MNIST (`experiment.py`) and produces the action‑probability / reward‑over‑time plots used for Figure 2a–b (see `./image/actions over time non cl.png`, `./image/actions over time cl.png`, and related files in `./image/plots/`),  
  - runs the baselines and data‑valuation comparison experiments, generating plots such as `dvn_vs_random_buffer.png` and `dvn_vs_random_generated.png` (Figure 2c,f), and  
  - runs the data‑value‑vs‑novelty analysis, producing correlation plots such as `pearson_r_by_stage_distance_buffer.png` and `pearson_r_by_stage_distance_gen.png` in `./image/plots/` (Figure 2d,g).
* For MNIST results to corroborate the findings, change the dataset from FashionMNIST to MNIST in `./image/environment.py` and re‑run the notebook (or at least the cells that call `data_value_vs_novelty.py`).

#### Figure 3 (maze task):
* Open `./maze/Main experiment.ipynb` and run all cells.  
  This:
  - trains the maze meta‑controller via `experiment.py` and produces the action‑probability / reward‑over‑time plot used for Figure 3a (saved as `./maze/actions over time.png`),  
  - runs incremental action‑selection baselines for comparison, and  
  - runs `stage_path_change_analysis.py` to generate the path‑change and value‑by‑location plots used in Figure 3 (e.g. `./maze/plots/stage_path_change_bias.png`, `./maze/plots/stage_path_length_values.png`, `./maze/plots/stage_path_value_summary.png`, `./maze/plots/stage_path_manhattan_values.png`).

#### Figure 4 (graph task):
* Open `./graph/Main experiment.ipynb` and run all cells.  
  The notebook:
  - calls `metacontroller.py` to train the graph meta‑controller on both the spatial and family‑tree builders, and  
  - calls `recreate_plots.py` to regenerate the main figures from the saved logs.  
  The resulting plots are saved as `./graph/SpatialBuilder_results.png` and `./graph/FamilyTreeBuilder_results.png`, which contain the action‑probability and reward‑over‑time curves and final action‑frequency summaries used in Figure 4.
