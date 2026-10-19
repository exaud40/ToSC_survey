"""
Hierarchical SCC-RA: sensing-computation-communication resource allocation for
Federated Edge Learning (FEL) in Task-Oriented Semantic Communication (ToSC).

Implements Algorithm 1 and Eqs. (1)-(9) of the manuscript (Section 3.3):

  * three-tier hierarchy: cloud server -> M edge servers -> N end devices
  * every device is an autonomous Q-learning agent (epsilon-greedy policy,
    replay buffer, Q-table) choosing a_i(t) = (c_i(t), b_i(t))
  * local SGD (Eqs. 2-3), edge aggregation (weights n_i / N_m) and cloud
    aggregation (weights N_m / N)
  * reward R(s, a) = lam1 * f_conv - lam2 * C_latency - lam3 * C_accuracy (Eq. 9)
  * exploration decay  epsilon <- alpha * epsilon

Only NumPy is required. The learning task is a synthetic non-IID logistic
regression problem so the script runs anywhere; swap `make_data` / `local_sgd`
for your own model and dataset to use it on a real ToSC task.

Modelling assumptions (not fixed by the manuscript, documented here):
  * Q-learning uses a separate step size `q_lr`; the manuscript's `alpha`
    is used as the epsilon decay factor, as written in Algorithm 1.
  * Channel gain ~ Rayleigh fading, sensor accuracy level ~ uniform.
  * Latency is the normalised sum of computation and communication delay.

Usage:
    python scc_ra.py --devices 12 --edges 3 --rounds 30
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration (inputs of Algorithm 1)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    N: int = 12                 # number of end devices
    M: int = 3                  # number of edge servers
    tau: int = 3                # edge aggregation rounds per global round
    local_steps: int = 5        # SGD steps per edge round
    R: int = 30                 # global communication rounds
    alpha: float = 0.95         # epsilon decay factor (Algorithm 1)
    gamma: float = 0.9          # discount factor
    q_lr: float = 0.1           # Q-learning step size
    buffer_size: int = 5000     # replay buffer D
    B: int = 32                 # replay minibatch size
    C_max: float = 1.0          # max computational resource per device
    W_max: float = 1.0          # max bandwidth per edge cluster (normalised)
    lam1: float = 1.0           # reward weights (Eq. 9)
    lam2: float = 0.5
    lam3: float = 1.0
    acc_threshold: float = 0.6  # minimum sensing accuracy requirement
    dim: int = 10               # feature dimension of the learning task
    lr: float = 0.2             # SGD learning rate eta
    samples_per_device: int = 400
    seed: int = 0


N_LEVELS = 3  # discretisation of channel / bandwidth / sensor state and actions


# --------------------------------------------------------------------------- #
# Synthetic non-IID learning task
# --------------------------------------------------------------------------- #
def make_data(cfg: Config, rng: np.random.Generator):
    w_true = rng.normal(size=cfg.dim)
    devices = []
    for i in range(cfg.N):
        shift = rng.normal(scale=1.0, size=cfg.dim)          # feature skew (non-IID)
        X = rng.normal(size=(cfg.samples_per_device, cfg.dim)) + shift
        y = (X @ w_true + rng.normal(scale=0.5, size=len(X)) > 0).astype(float)
        devices.append((X, y))
    Xt = rng.normal(size=(2000, cfg.dim))
    yt = (Xt @ w_true > 0).astype(float)
    return devices, (Xt, yt)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def loss_fn(w, X, y):
    p = sigmoid(X @ w)
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def accuracy(w, X, y):
    return float(np.mean((sigmoid(X @ w) > 0.5) == y))


def local_sgd(w, X, y, steps, batch, lr, rng):
    """Eqs. (2)-(3): w <- w - eta * g_i(w), g_i = minibatch stochastic gradient."""
    w = w.copy()
    for _ in range(steps):
        idx = rng.integers(0, len(X), size=batch)
        g = X[idx].T @ (sigmoid(X[idx] @ w) - y[idx]) / batch
        w -= lr * g
    return w


# --------------------------------------------------------------------------- #
# Q-learning agent (one per end device)
# --------------------------------------------------------------------------- #
class Agent:
    """State  s = (channel level, bandwidth level, sensor-accuracy level).
    Action a = (compute level, bandwidth level)."""

    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg, self.rng = cfg, rng
        self.n_states = N_LEVELS ** 3
        self.n_actions = N_LEVELS ** 2
        self.Q = np.random.default_rng(rng.randint(0, 10**6)).normal(
            scale=0.01, size=(self.n_states, self.n_actions))
        self.buffer: deque = deque(maxlen=cfg.buffer_size)

    @staticmethod
    def state_index(ch, bw, sens):
        return (ch * N_LEVELS + bw) * N_LEVELS + sens

    def act(self, s, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) < self.cfg.B:
            return
        for s, a, r, s2 in self.rng.sample(list(self.buffer), self.cfg.B):
            target = r + self.cfg.gamma * np.max(self.Q[s2])   # Bellman target
            self.Q[s, a] += self.cfg.q_lr * (target - self.Q[s, a])


def decode_action(a):
    return a // N_LEVELS, a % N_LEVELS   # (compute level, bandwidth level)


# --------------------------------------------------------------------------- #
# Hierarchical SCC-RA (Algorithm 1)
# --------------------------------------------------------------------------- #
def level_of(x, lo, hi):
    return int(np.clip((x - lo) / (hi - lo + 1e-12) * N_LEVELS, 0, N_LEVELS - 1))


def run(cfg: Config):
    nprng = np.random.default_rng(cfg.seed)
    pyrng = random.Random(cfg.seed)
    devices, (Xt, yt) = make_data(cfg, nprng)

    clusters = [list(range(m, cfg.N, cfg.M)) for m in range(cfg.M)]   # N_m
    agents = [Agent(cfg, pyrng) for _ in range(cfg.N)]
    w_global = np.zeros(cfg.dim)                                        # w^(0)
    epsilon = 1.0
    bw_level = [N_LEVELS - 1] * cfg.N

    def observe(i):
        h = nprng.rayleigh(scale=1.0)                  # channel gain
        sens = nprng.uniform(0.4, 1.0)                 # sensor accuracy level
        return h, sens, Agent.state_index(level_of(h, 0.0, 2.0), bw_level[i],
                                          level_of(sens, 0.4, 1.0))

    obs = {i: observe(i) for i in range(cfg.N)}
    history = []

    for g in range(1, cfg.R + 1):                       # global communication rounds
        edge_models, edge_sizes = [], []
        for m, members in enumerate(clusters):          # edge servers (parallel)
            w_edge = w_global.copy()                    # cloud -> edge broadcast
            for t in range(cfg.tau):                    # edge aggregation rounds
                # ---- action selection + resource allocation (c_i <= C_max, sum b_i <= W_max)
                acts, c_alloc, b_alloc = {}, {}, {}
                for i in members:
                    s = obs[i][2]
                    acts[i] = agents[i].act(s, epsilon)
                    cl, bl = decode_action(acts[i])
                    c_alloc[i] = cfg.C_max * (cl + 1) / N_LEVELS
                    b_alloc[i] = (bl + 1) / N_LEVELS
                total_b = sum(b_alloc.values())
                if total_b > cfg.W_max:                 # enforce bandwidth budget
                    for i in members:
                        b_alloc[i] *= cfg.W_max / total_b

                local_models, sizes = [], []
                for i in members:
                    X, y = devices[i]
                    h, sens, s = obs[i]
                    batch = int(8 + 56 * c_alloc[i] / cfg.C_max)       # compute -> batch size
                    loss_before = loss_fn(w_edge, X, y)
                    w_i = local_sgd(w_edge, X, y, cfg.local_steps, batch, cfg.lr, nprng)
                    f_conv = 10.0 * (loss_before - loss_fn(w_i, X, y))   # convergence progress

                    # latency model: computation + communication (normalised)
                    t_comp = cfg.local_steps * batch / (c_alloc[i] * 200.0 + 1e-9)
                    rate = b_alloc[i] * np.log2(1.0 + 10.0 * h ** 2) + 1e-9
                    t_comm = 1.0 / rate
                    c_latency = t_comp + t_comm

                    sens_acc = sens * (0.7 + 0.3 * c_alloc[i] / cfg.C_max)
                    c_accuracy = max(0.0, cfg.acc_threshold - sens_acc)

                    reward = cfg.lam1 * f_conv - cfg.lam2 * c_latency - cfg.lam3 * c_accuracy  # Eq. (9)

                    bw_level[i] = decode_action(acts[i])[1]
                    obs[i] = observe(i)
                    agents[i].store((s, acts[i], reward, obs[i][2]))
                    agents[i].update()

                    local_models.append(w_i)
                    sizes.append(len(X))

                # ---- edge aggregation: w_m = sum_i (n_i / N_m) w_i
                sizes = np.array(sizes, dtype=float)
                w_edge = sum(w * (n / sizes.sum()) for w, n in zip(local_models, sizes))
            edge_models.append(w_edge)
            edge_sizes.append(sum(len(devices[i][0]) for i in members))

        # ---- cloud aggregation: w^(r+1) = sum_m (N_m / N) w_m
        edge_sizes = np.array(edge_sizes, dtype=float)
        w_global = sum(w * (n / edge_sizes.sum()) for w, n in zip(edge_models, edge_sizes))
        epsilon *= cfg.alpha                              # decay exploration

        acc = accuracy(w_global, Xt, yt)
        history.append(acc)
        if g == 1 or g % 5 == 0 or g == cfg.R:
            print(f"round {g:3d} | epsilon {epsilon:.3f} | global test acc {acc:.4f}")

    # greedy policies pi*_c, pi*_b per agent / state
    policies = []
    for ag in agents:
        pol = {}
        for s in range(ag.n_states):
            cl, bl = decode_action(int(np.argmax(ag.Q[s])))
            pol[s] = (cfg.C_max * (cl + 1) / N_LEVELS, (bl + 1) / N_LEVELS)
        policies.append(pol)
    return policies, w_global, history


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical SCC-RA for FEL (Algorithm 1)")
    d = Config()
    for name in ("N", "M", "tau", "R", "local_steps", "seed"):
        p.add_argument(f"--{name}", type=int, default=getattr(d, name))
    for name in ("alpha", "gamma", "lr", "lam1", "lam2", "lam3"):
        p.add_argument(f"--{name}", type=float, default=getattr(d, name))
    # friendlier aliases
    p.add_argument("--devices", type=int, dest="N")
    p.add_argument("--edges", type=int, dest="M")
    p.add_argument("--rounds", type=int, dest="R")
    return p.parse_args()


if __name__ == "__main__":
    args = vars(parse_args())
    cfg = Config(**{k: v for k, v in args.items() if v is not None})
    policies, w_final, hist = run(cfg)
    print(f"\nFinal global model accuracy: {hist[-1]:.4f}")
    print(f"Learned resource-allocation policies for {len(policies)} agents "
          f"(state -> (compute, bandwidth)).")
