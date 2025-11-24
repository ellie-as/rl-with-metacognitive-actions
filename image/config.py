import torch


class Config:
    SEED = 42
    BUFFER_CAPACITY = 200
    #STAGES = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    STAGES = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    NUM_TO_SELECT = 50
    VAL_SET_SIZE = 1000
    NOISE_FRACTION = 0.1
    SHAPLEY = True
    MC_ITERS = 10
    TRUE_SHAPLEY_SUBSET = 0.25
    NUM_CLASSES = 10
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    MMR = True
    LEVELS = (0.25, 0.5, 0.75, 1.0)
    PARTIAL_FIT_EPS = 100
    LAMBDA_PARAM = 0.9
    STEPS_PER_EPISODE = 3
    NOISY_TEST = True
    # Dataset selector: "fashion_mnist" or "mnist"
    DATASET = "fashion_mnist"
