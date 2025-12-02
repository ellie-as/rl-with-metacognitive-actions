class Config:
    # Meta-controller parameters
    SEED = 123
    EVALUATION_EPISODES = 100       # episodes used to evaluate agent before / after each step
    GRID_SIZE = 6                  # width / height of maze
    # Maze generation mode: 'default' or 'extra_links'.
    # - default: existing BFS-connected barrier generation
    # - extra_links: after full reset generation, reopen a fraction of remaining barriers to create multiple paths
    MAZE_MODE = 'extra_links'
    EXTRA_LINKS_REOPEN_FRACTION = 0.15  # fraction of remaining barrier cells reopened when MAZE_MODE='extra_links'
    CONSTANT_EP_LENGTH = True      # whether every episode is the same length
    RESET_INTERVAL = 3             # reset maze every RESET_INTERVAL if CONSTANT_EP_LENGTH is True
    P_CHANGE = 0.1                 # otherwise, reset maze with prob P_CHANGE at each step
    INITIAL_TRAINING_STEPS = 0 # steps to train with normal .learn() at episode start
    WORLD_MODEL_TYPE = 'cache'        # 'nn' (neural net), 'cache' (tabular replay), 'debug' (cache with full coverage)

    # Shared agent exploration/discount (used by DQN and valuation helpers)
    Q_DISCOUNT = 0.95
    Q_EPSILON = 0.2
    Q_EPSILON_DECAY = 0.997
    Q_MIN_EPSILON = 0.05
    FIX_EPSILON = True        # when True, keep exploration epsilon fixed at Q_EPSILON

    # Data valuation parameters
    VALUATION_TYPE = 'approx_shapley' #'shapley'      # start-goal values use Shapley estimation
    MINI_TRAIN_STEPS = 1000         # number of training steps for getting real values
    SG_TRAIN_EPOCHS = 200          # epochs to train the start-goal value estimator
    SG_ESTIMATOR_MODEL = 'forest_with_history'     # 'knn', 'linear', 'forest', or 'forest_with_history'
    SG_FOREST_TREES = 100          # Used when SG_ESTIMATOR_MODEL='forest'
    INCLUDE_IMPOSSIBLE_GOALS = False  # when True, candidate set includes blocked goal cells
    SG_HISTORY_MIN_NEW_FRACTION = 0.25  # Minimum fraction of weight assigned to latest Shapley batch
    MODE = 'top_with_mmr'         # options: top_with_mmr, bottom_with_mmr, top, bottom, longest_paths, or random
    LAMBDA_PARAM = 0.9 
    NUM_PERMUTATIONS = 10          # permutations used for Shapley estimation
    MASK_ILLEGAL = False            # when True, mask illegal actions everywhere; when False, never mask
    VALUATION_MAX_WORKERS = 1    # Optional cap on parallel Shapley workers; set to 1 to disable threading

    # Maze incremental reset behaviour
    INCREMENTAL_UPDATE_STRATEGY = 'centrality'  # options: 'centrality', 'random'
    # Agent selection (DQN only)
    AGENT_TYPE = 'sb3_dqn'
    DQN_HIDDEN_SIZE = 128
    DQN_LEARNING_RATE = 1e-4
    SB3_DQN_KWARGS = None  # Optional dict passed to stable-baselines3 DQN when AGENT_TYPE='sb3_dqn'

    # Episode-based training (hippocampus and bursts)
    HIPPOCAMPUS_REAL_EPISODES = 50   # number of real episodes to encode
    EPISODES_PER_BURST = 200         # number of episodes per real/dream training burst
    EPISODE_MAX_STEPS = 50           # max steps per stored episode
    TOP_TRAJECTORIES = EPISODES_PER_BURST          # how many top-valued start-goal pairs to use in dream
    NUM_TO_ESTIMATE = HIPPOCAMPUS_REAL_EPISODES         # number of pairs to estimate Shapley values on
    DREAM_EPISODE_REPEATS = 5        # number of times to repeat each selected dream episode
