# Task-Oriented Semantic Communication (ToSC)

This is the survey **"Towards Task-Oriented Semantic Communication: A Survey on Technologies, Opportunities, and Challenges"** by Exaud Tarimo, Huanlai Xing, Jincheng Peng, Li Feng, Lexi Xu, and Amir Rehman (Southwest Jiaotong University / China United Network Communications Corporation).

Task-Oriented Semantic Communication (ToSC) reduces communication cost by transmitting only the information a downstream task needs, instead of reconstructing raw signals bit-for-bit. The survey covers theoretical foundations, system architectures (end-to-end, multi-user, distributed), resource optimization (edge inference, federated edge learning), benchmark comparisons, and applications. This repository provides runnable reference code for the two algorithmic building blocks presented in the paper.

## Repository contents

| File | Paper reference | What it implements |
|------|-----------------|--------------------|
| `tosc_vib.py` | Sec. 2.1, 2.7, 3.1; Fig. 3, Fig. 8; Table 6 | Multi-view variational information bottleneck (VIB) semantic encoder with an AWGN channel in the training loop and a task decoder at the edge server |
| `scc_ra.py` | Sec. 3.3; Algorithm 1; Eqs. (1)-(9); Fig. 11 | Hierarchical sensing-computation-communication resource allocation (SCC-RA) for federated edge learning (cloud, edge servers, end devices) using multi-agent Q-learning |

### `tosc_vib.py`: task-oriented feature encoding

Each device observes one view `x_k`, extracts features, and produces a Gaussian latent `z_k = mu_k + sigma_k * eps`. The code is power-normalised, sent over an AWGN channel, and the edge server concatenates all received `z_k` for task inference. Training minimises the IB loss from Table 6:

```
L = CE(y, y_hat) + beta * KL( q(z|x) || N(0, I) )
```

- `beta` controls the compression/accuracy trade-off (minimal-sufficient features).
- The channel is part of training (noise-aware E2E optimization), giving graceful degradation at low SNR.
- `--beta-sweep` prints accuracy against a KL rate proxy for several values of `beta`.

### `scc_ra.py`: hierarchical resource allocation for FEL

Follows Algorithm 1: every end device is an autonomous agent that selects `a_i(t) = (c_i(t), b_i(t))` (compute and bandwidth) with an epsilon-greedy policy, a Q-table, and a replay buffer. Local SGD (Eqs. 2-3) is followed by edge aggregation (weights `n_i / N_m`) and cloud aggregation (weights `N_m / N`). The reward is Eq. (9):

```
R(s, a) = lam1 * f_conv - lam2 * C_latency - lam3 * C_accuracy
```

Exploration decays as `epsilon <- alpha * epsilon`. The script outputs the learned allocation policies and the final global model.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install numpy torch
```

`scc_ra.py` needs only NumPy. `tosc_vib.py` needs PyTorch (CPU is enough).

## Usage

```bash
# Multi-view VIB semantic encoder
python tosc_vib.py --views 3 --beta 1e-3 --train-snr 10
python tosc_vib.py --beta-sweep

# Hierarchical SCC-RA for federated edge learning
python scc_ra.py --devices 12 --edges 3 --rounds 30
```

Run either script with `-h` for all options.

## Notes and limitations

- Both scripts use **synthetic data** so they run with no downloads. They demonstrate the algorithms; they do not reproduce the benchmark numbers in the survey's tables and figures (ImageNet, CIFAR, LibriSpeech, Europarl, etc.). Replace `make_multiview` (in `tosc_vib.py`) or `make_data` / `local_sgd` (in `scc_ra.py`) with your own dataset and model.
- `scc_ra.py` uses tabular Q-learning with discretised states and actions. The channel, sensing, and latency models are simple normalised assumptions documented in the file header, because the manuscript does not fix them.
- In Algorithm 1, `alpha` is the epsilon decay factor; the Q-learning step size is a separate parameter (`q_lr`).

## Research roadmap (from the survey's outlook)

The survey identifies the following open directions, which this codebase is meant to support experimentation toward:

1. **Foundation-model semantic encoders**: integrating large multimodal foundation models into semantic encoders/decoders for zero-shot task adaptation, and bridging them with the IB principle (contrastive alignment and masked modeling as approximations of VIB) for adaptive semantic compression.
2. **Continual learning on devices**: accumulating new task knowledge and discarding outdated information using parameter-efficient fine-tuning, federated continual learning, and semantic knowledge distillation.
3. **Cooperative multi-agent reasoning**: exchanging contextual and causal explanations, not just observations, in representations that are interpretable and robust to partial observability and channel noise.
4. **AI-native 6G co-design**: cross-layer design with reconfigurable intelligent surfaces (RIS) and digital-twin-assisted optimization so encoders adapt to channel state and topology.
5. **Unequal error protection and adaptive JSCC**: prioritising task-critical features via adaptive power allocation or coding redundancy, and RL-based schedulers coordinating compression with cross-layer resources in multi-user, multi-task settings.
6. **Embodied AI agents**: control-aware semantic metrics, stability-guaranteed semantic adaptation, and analysis of how semantic distortion affects closed-loop control (autonomous vehicles, UAV swarms, collaborative robots).
7. **Standardised benchmarks** covering modalities, SNRs, and inference tasks, so ToSC methods can be compared fairly.

The paper also anticipates that LLM-powered encoders may transmit compressed reasoning states, intent representations, or task-conditioned latent tokens rather than raw features.

## Citation

If you use this code, please cite the survey (update with final publication details):

```bibtex
@article{tarimo_tosc_survey,
  title   = {Towards Task-Oriented Semantic Communication: A Survey on Technologies, Opportunities, and Challenges},
  author  = {Tarimo, Exaud and Xing, Huanlai and Peng, Jincheng and Feng, Li and Xu, Lexi and Rehman, Amir},
  journal = {TODO: venue},
  year    = {TODO}
}
```

## Contact

Corresponding author: Huanlai Xing (hxx@home.swjtu.edu.cn), School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, China.

## Acknowledgment

Supported in part by the National Natural Science Foundation of China (No. 62172342 and No. U23A20273) and the Fundamental Research Funds for the Central Universities, P. R. China.

## License

TODO: add a license (e.g., MIT or Apache-2.0) before publishing.
