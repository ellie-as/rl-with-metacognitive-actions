import logging
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import defaultdict, deque
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# Mixture-of-Gaussians Generator with Binarisation
class GaussianGenerator:
    def __init__(self, n_clusters=3):
        # Number of clusters to use for each class
        self.n_clusters = n_clusters
        # Dictionary mapping label -> (kmeans, cluster_params)
        # cluster_params is a dict: cluster_id -> (mean, cov, cluster_size)
        self.class_cluster_params = {}

    def fit(self, buffer_samples):
        """Update mixture model for classes present in the buffer using KMeans"""
        class_data = defaultdict(list)
        for img, label in buffer_samples:
            img_flat = img.flatten()  # shape: (784,)
            class_data[label].append(img_flat)

        for label, imgs in class_data.items():
            imgs = np.stack(imgs)  # shape: (N,784)
            # Use at most n_clusters (or fewer if not enough samples)
            n_clusters = min(self.n_clusters, imgs.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(imgs)
            cluster_params = {}
            for cluster in range(n_clusters):
                cluster_imgs = imgs[cluster_labels == cluster]
                if cluster_imgs.shape[0] < 2:
                    continue
                mean = cluster_imgs.mean(axis=0).astype(np.float32)
                cov = np.cov(cluster_imgs, rowvar=False)
                cov += np.eye(imgs.shape[1]) * 1e-6  # stabilize covariance
                try:
                    chol = np.linalg.cholesky(cov).astype(np.float32)
                except np.linalg.LinAlgError:
                    # fall back to diagonal noise if covariance is singular
                    diag = np.sqrt(np.clip(np.diag(cov), 1e-6, None)).astype(np.float32)
                    chol = np.diag(diag)
                cluster_params[cluster] = (mean, chol, cluster_imgs.shape[0])
            self.class_cluster_params[label] = (kmeans, cluster_params)

    def generate(self, label, num_samples=1):
        """Generate samples for a given class using the mixture model.
           The generated samples are binarised with a threshold of 0.5.
        """
        if label not in self.class_cluster_params:
            raise ValueError(f"No data for class {label}")
        kmeans, cluster_params = self.class_cluster_params[label]
        clusters = list(cluster_params.keys())
        sizes = np.array([cluster_params[c][2] for c in clusters])
        weights = sizes / sizes.sum()
        counts = np.random.multinomial(num_samples, weights)
        sampled_batches = []
        for cluster, count in zip(clusters, counts):
            if count == 0:
                continue
            mean, chol, _ = cluster_params[cluster]
            noise = np.random.standard_normal(size=(count, mean.shape[0])).astype(np.float32)
            draws = noise @ chol.T + mean
            sampled_batches.append(draws)
        if not sampled_batches:
            raise ValueError(f"No valid clusters to sample for class {label}")
        samples = np.concatenate(sampled_batches, axis=0)
        if samples.shape[0] != num_samples:
            # adjust by repeating the last batch if numerical issues trimmed size
            deficit = num_samples - samples.shape[0]
            if deficit > 0 and sampled_batches:
                extra = sampled_batches[-1][:deficit]
                samples = np.concatenate([samples, extra], axis=0)
        samples = samples[:num_samples]
        np.random.shuffle(samples)
        # Binarise samples at threshold 0.5
        samples = (samples >= 0.5).astype(np.float32)
        samples = samples.reshape(-1, 1, 28, 28)
        images = torch.from_numpy(samples)
        labels = torch.full((num_samples,), label, dtype=torch.long)
        return images, labels

    def plot_samples(self, samples_per_class=5):
        """Plot generated samples for each known class using the mixture model"""
        known_classes = sorted(self.class_cluster_params.keys())
        if not known_classes:
            print("No classes trained yet!")
            return

        n_classes = len(known_classes)
        fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(12, 15))
        fig.suptitle("Generated Samples by Class (Mixture of Gaussians with Binarisation)", fontsize=14, y=1.02)

        for i, label in enumerate(known_classes):
            images, _ = self.generate(label, num_samples=samples_per_class)
            for j in range(samples_per_class):
                ax = axes[i, j] if n_classes > 1 else axes[j]
                ax.imshow(images[j][0].squeeze(), cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_title(f"Class {label}", loc='left')

        plt.tight_layout()
        plt.show()
