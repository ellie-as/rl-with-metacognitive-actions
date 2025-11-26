import logging
import random
from collections import defaultdict, deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from gymnasium import spaces
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from random import randrange
from classifier import Classifier 
from config import *
from generator import GaussianGenerator
from valuator import train_value_estimator_with_shapley
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.spatial.distance import cdist
import torch
from torchvision import datasets, transforms
import numpy as np
import random

class SaltPepper:
    """
    Apply salt-and-pepper noise to a tensor image in [0,1].

    On every call we draw p ∈ Config.LEVELS with equal probability and
    corrupt exactly p of the pixels.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img is 1×H×W float tensor in [0,1]
        p = random.choice(Config.LEVELS)
        rnd = torch.rand_like(img)
        salt = rnd <  p / 2      
        pepper = rnd > 1 - p / 2
        img = img.clone()
        img[salt]   = 0.0 # switch to 1 for white pixels
        img[pepper] = 0.0
        return img

class MNISTStageLoader:
    """
    Loads Fashion-MNIST once and then serves slices according to Config.STAGES,
    with salt-and-pepper noise applied to **every** image (train/val/test).
    """
    def __init__(self):
        # ToTensor  → SaltPepper  → return noisy tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            SaltPepper(),
        ])

        if Config.NOISY_TEST:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                SaltPepper(),
            ])
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        root = "./data"
        if getattr(Config, "DATASET", "fashion_mnist").lower() == "mnist":
            self.train_dataset = datasets.MNIST(
                root, train=True, download=True, transform=transform
            )
            self.test_val_dataset = datasets.MNIST(
                root, train=False, download=True, transform=test_transform
            )
        else:
            self.train_dataset = datasets.FashionMNIST(
                root, train=True, download=True, transform=transform
            )
            self.test_val_dataset = datasets.FashionMNIST(
                root, train=False, download=True, transform=test_transform
            )

        val_size   = Config.VAL_SET_SIZE
        test_size = len(self.test_val_dataset) - val_size
        self.test_dataset, self.val_dataset = torch.utils.data.random_split(
            self.test_val_dataset,
            [test_size, val_size],
            generator=torch.Generator().manual_seed(Config.SEED),
        )

    def get_stage_samples(self, stage: int, num_samples: int):
        stage_digits = Config.STAGES[stage]
        samples = []
        while len(samples) < num_samples:
            idx = random.randint(0, len(self.train_dataset) - 1)
            img, label = self.train_dataset[idx]
            if label in stage_digits:
                samples.append((img.numpy(), label))
        return samples[:num_samples]


class MetaLearningEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.data_loader = MNISTStageLoader()
        self.current_stage = 0
        self.buffer = deque(maxlen=Config.BUFFER_CAPACITY)
        self.classifier = Classifier(verbose=0, train_split=False)
        self.value_estimator = SGDRegressor() #KNeighborsRegressor(n_neighbors=3)
        self.generator = GaussianGenerator(n_clusters=5)

        self.class_acc = np.zeros(10, dtype=np.float32)

        self._initialize_buffer()

        self.action_space = spaces.Discrete(3)
        # Observation contains buffer class counts, training step index, and previous meta‑action.
        self.observation_space = spaces.Dict(
            {
                # Per‑class counts of items currently in the buffer.
                "buffer_class_counts": spaces.Box(
                    low=0,
                    high=Config.BUFFER_CAPACITY,
                    shape=(10,),
                    dtype=np.float32,
                ),
                "time_since_last_train": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                # -1 denotes "no previous action" at episode start.
                "last_action": spaces.Box(
                    low=-1,
                    high=self.action_space.n - 1,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.val_loader = DataLoader(self.data_loader.val_dataset, batch_size=Config.NUM_TO_SELECT, shuffle=True)
        self.test_loader = DataLoader(self.data_loader.test_dataset, batch_size=Config.NUM_TO_SELECT, shuffle=True)

        self.last_performance = 0.0
        self.time_since_train = 0

        # test-mode helpers
        self.test_mode = False
        self.forced_actions = []
        self.current_action_idx = 0
        # store previous meta‑action for the controller's observation
        self.last_action = -1

    def set_test_mode(self, action_sequence):
        """
        Enable test mode with a forced sequence of actions.
        """
        self.test_mode = True
        self.forced_actions = list(action_sequence)
        self.current_action_idx = 0

    def _print_class_distribution(self, samples, source_name):
        counts = defaultdict(int)
        for _, lbl in samples:
            counts[int(lbl)] += 1
        total = len(samples)
        if total:
            dist = ", ".join([f"{cls}: {cnt/total:.1%}" for cls, cnt in sorted(counts.items())])
            print(f"Class distribution from {source_name} ⇒ {dist}")

    def _initialize_buffer(self):
        samples = self.data_loader.get_stage_samples(self.current_stage, Config.BUFFER_CAPACITY)
        for img_np, lbl in samples:
            self.buffer.append((img_np, lbl))

    def _get_observation(self):
        counts = np.zeros(10, dtype=np.float32)
        for _, lbl in self.buffer:
            counts[int(lbl)] += 1
        # Convert counts to fractions so the vector reflects the class distribution
        # of the current hippocampal buffer rather than raw counts.
        total = float(len(self.buffer))
        if total > 0.0:
            counts /= total
        return {
            "buffer_class_counts": counts,
            "time_since_last_train": np.array(
                [self.time_since_train], dtype=np.float32
            ),
            "last_action": np.array([self.last_action], dtype=np.float32),
        }
    
    def _mmr_select_indices(
            self,
            X_flat: np.ndarray,
            vals:   np.ndarray,
            k: int,
            *,
            labels=None,
            max_per_class=None,
            balance_classes=True):
        """
        MMR selection with min–max normalisation of both relevance and diversity.
    
        Args are identical to the original _mmr_select_indices.
        """
        λ = Config.LAMBDA_PARAM
        n = len(vals)
        assert X_flat.shape[0] == n
    
        # 1) Normalise relevance scores to [0,1]
        v_min, v_max = float(vals.min()), float(vals.max())
        rel_norm = np.zeros_like(vals, dtype=np.float32) if v_max == v_min \
                   else (vals - v_min) / (v_max - v_min)
    
        # 2) Pre-compute and normalise pair-wise Euclidean distances
        D = cdist(X_flat, X_flat, metric="euclidean")    # (n × n) matrix
        d_min, d_max = float(D.min()), float(D.max())
        if d_max == d_min:
            D_norm = np.zeros_like(D, dtype=np.float32)
        else:
            D_norm = (D - d_min) / (d_max - d_min)
    
        # 3) Greedy MMR with optional class balancing
        candidates   = set(range(n))
        selected     = []
        class_counts = defaultdict(int)
    
        if balance_classes and labels is not None:
            quota = max_per_class if max_per_class is not None \
                    else math.ceil(k / len(np.unique(labels)))
        else:
            quota = None
    
        while len(selected) < k and candidates:
            best_i, best_score = None, -np.inf
            for i in candidates:
                if balance_classes and labels is not None and quota is not None:
                    if class_counts[labels[i]] >= quota:
                        continue
    
                # Relevance term (already normalised)
                rel = rel_norm[i]
    
                # Diversity term: min normalised distance to the current set
                if not selected:
                    score = rel # first pick ignores diversity
                else:
                    div = D_norm[i, selected].min()
                    score = λ * rel + (1 - λ) * div
    
                if score > best_score:
                    best_score, best_i = score, i
    
            if best_i is None: 
                break
            selected.append(best_i)
            candidates.remove(best_i)
            if balance_classes and labels is not None and quota is not None:
                class_counts[labels[best_i]] += 1
    
        return selected

    def _partial_fit(self, X, y, epochs=3):
        y = y.astype(np.int64)
        if not hasattr(self.classifier, "initialized_"):
            self.classifier.partial_fit(X, y, classes=np.arange(10))
            epochs -= 1
        for _ in range(epochs):
            self.classifier.partial_fit(X, y)

    def _train_classifier_on_buffer(self):
        """Fine-tune the classifier using samples from the buffer."""
        if not self.buffer:
            return

        # flat features and labels
        X_flat = np.array([x.flatten() for x, _ in self.buffer])
        yb     = np.array([lbl for _, lbl in self.buffer], dtype=np.int64)
        k      = min(Config.NUM_TO_SELECT, len(self.buffer))

        # compute a value for every sample
        if Config.SHAPLEY:
            buf = list(self.buffer); random.shuffle(buf)
            self.value_estimator = train_value_estimator_with_shapley(
                self.value_estimator, buf, self.classifier,
                self.data_loader.val_dataset
            )
            one_hot = np.eye(Config.NUM_CLASSES, dtype=np.float32)[yb]
            vals    = self.value_estimator.predict(
                          np.concatenate([X_flat, one_hot], axis=1))
        else:
            rng  = np.random.default_rng(self.time_since_train)
            vals = rng.random(len(self.buffer))          # uniform[0,1]

        # choose indices according to config flags
        if Config.MMR:
            sel_inds = self._mmr_select_indices(X_flat, vals, k, labels=yb)
        else:
            # take the k highest-valued samples
            sel_inds = np.argsort(vals)[-k:][::-1]       # descending

        chosen = [self.buffer[i] for i in sel_inds]
        self._print_class_distribution(chosen, "buffer (selected)")

        # fine-tune classifier
        Xb = np.stack([x for x, _ in chosen]).astype(np.float32)
        yb = np.array([lbl for _, lbl in chosen], dtype=np.int64)
        self._partial_fit(Xb, yb, epochs=Config.PARTIAL_FIT_EPS)
        _, per_class_acc = self.classifier.evaluate_with_breakdown(self.val_loader)
        self.class_acc = per_class_acc

    def _train_on_generated(self):
        """Fine-tune classifier on samples drawn from the generator."""
        if not getattr(self.generator, 'class_cluster_params', None):
            logging.warning("Generator not trained – skipping.")
            return

        classes = [
            c
            for c, (_, params) in sorted(self.generator.class_cluster_params.items())
            if params
        ]
        if not classes:
            logging.warning("Generator holds no clusters – skipping.")
            return

        if Config.MMR is False and Config.SHAPLEY is False:
            per = Config.NUM_TO_SELECT // max(1, len(classes))
        else:
            per = Config.BUFFER_CAPACITY // max(1, len(classes))
        Xg_list, yg_list = [], []
        print("Generating samples")
        for c in classes:
            imgs, labs = self.generator.generate(c, per)
            Xg_list.append(imgs); yg_list.append(labs)

        Xg      = torch.cat(Xg_list)
        yg      = torch.cat(yg_list)
        Xg_flat = Xg.view(Xg.size(0), -1).cpu().numpy()
        y_gen   = yg.cpu().numpy().astype(np.int64)
        k       = min(Config.NUM_TO_SELECT, len(y_gen))

        # compute value for each generated sample
        if Config.SHAPLEY:
            print("Getting Shapley values")
            buf = [(Xg[i].cpu().numpy(), int(yg[i])) for i in range(len(yg))]
            random.shuffle(buf)
            self.value_estimator = train_value_estimator_with_shapley(
                self.value_estimator, buf, self.classifier,
                self.data_loader.val_dataset
            )
            one_hot = np.eye(Config.NUM_CLASSES, dtype=np.float32)[y_gen]
            vals    = self.value_estimator.predict(
                         np.concatenate([Xg_flat, one_hot], axis=1))
        else:
            print("Getting dummy values")
            rng  = np.random.default_rng(self.time_since_train)
            vals = rng.random(len(y_gen))

        # choose indices
        if Config.MMR:
            print("Taking images using MMR procedure")
            sel_inds = self._mmr_select_indices(Xg_flat, vals, k,
                                                labels=y_gen)
        else:
            print("Taking images for top k dummy values")
            sel_inds = np.argsort(vals)[-k:][::-1]        # top-k values

        chosen = [(Xg[i], yg[i]) for i in sel_inds]
        self._print_class_distribution(chosen, "generated (selected)")

        # fine-tune classifier
        print("Finetuning classifier")
        Xb = np.stack([x for x, _ in chosen]).astype(np.float32)
        yb = np.array([int(lbl) for _, lbl in chosen], dtype=np.int64)
        self._partial_fit(Xb, yb, epochs=Config.PARTIAL_FIT_EPS)
        _, per_class_acc = self.classifier.evaluate_with_breakdown(self.val_loader)
        self.class_acc = per_class_acc
        
    def step(self, action):
        if self.test_mode:
            action = self.forced_actions[self.current_action_idx]
            self.current_action_idx += 1
        # record the action for next observation
        self.last_action = int(action)
        logging.info(f"Action {action} @ t={self.time_since_train}, stage={self.current_stage}")
        if action == 0:
            self._train_classifier_on_buffer()
        elif action == 1:
            self._train_generator()
        else:
            self._train_on_generated()
        self.time_since_train += 1
        obs = self._get_observation()
        done = self.time_since_train == Config.STEPS_PER_EPISODE
        
        if not done:
            # intermediate step → no reward
            return obs, 0.0, False, False, {}
        else:
            # otherwise, at final step:
            final_reward, per_class_acc = self.classifier.evaluate_with_breakdown(self.test_loader)
            self.class_acc = per_class_acc
            print(f"Obtained reward of {final_reward} at end of episode.")
            return obs, final_reward, True, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_stage = (self.current_stage + 1) % len(Config.STAGES)
        print(f"Progressing to stage {self.current_stage} of {len(Config.STAGES)}")
        self.buffer.clear()
        if self.current_stage == 0:
            print("Resetting classifier and generator")
            self.classifier = Classifier(verbose=0, train_split=False)
            self.generator = GaussianGenerator(n_clusters=5)
        self.last_performance = 0.0
        self.time_since_train = 0
        self.test_mode = False
        self.forced_actions = []
        self.current_action_idx = 0
        self.last_action = -1
        self._initialize_buffer()
        return self._get_observation(), {}

    def _train_generator(self):
        """Train Mixture-of-Gaussians Generator on current buffer samples"""
        if not self.buffer:
            logging.warning("Buffer empty - skipping generator training")
            return

        self.generator.fit(list(self.buffer))
        logging.info("Updated Mixture-of-Gaussians Generator with current buffer samples")
