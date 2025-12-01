# 🛰️ Online Learning Adaptive Beam Switching for 6G Networks

Official implementation of:

> **Online Learning-based Adaptive Beam Switching for 6G Networks: Enhancing Efficiency and Resilience**  
> *Seyed Bagher Hashemi Natanzi, Zhicong Zhu, Bo Tang*  
> [arXiv:2505.08032](https://arxiv.org/abs/2505.08032)

*This repository implements the online learning-based adaptive beam switching framework proposed in [arXiv:2505.08032](https://arxiv.org/abs/2505.08032).  
It features a GRU-based Deep Q-Learning (DQL) agent with Prioritized Experience Replay (PER) for real-time beam selection in dynamic 6G environments characterized by user mobility and time-correlated blockage.  
Compared to heuristic and Multi-Armed Bandit (MAB) baselines, the proposed method significantly improves throughput, SNR stability, and resilience, as demonstrated through detailed simulations using the Sionna platform.*

## 📁 Project Structure

The codebase follows Clean Code principles with a modular structure:

```
project_root/
├── src/
│   ├── __init__.py
│   ├── config.py          # All constants and Reward Profiles (Vanilla/Stability)
│   ├── models.py          # PyTorch Neural Networks (DQN)
│   ├── core.py            # ReplayBuffer and StateHistory (Logic for Ablation)
│   ├── environment.py     # Channel, Mobility, Beamforming physics
│   └── baselines.py       # Greedy and MAB-UCB classes
├── main.py                # The entry point (CLI arguments & Training Loop)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Docker Setup

```bash
docker build -t test-build .
docker run --gpus '"device=0"' --shm-size=8g -it --rm -v $(pwd):/workspace test-build
```

### Basic Usage

Run with default balanced mode:
```bash
python main.py
```

Run with vanilla baseline (Reviewer 2 requirement):
```bash
python main.py --mode vanilla --runs 5
```

Run ablation study (Reviewer 1 & 3 requirement):
```bash
python main.py --mode balanced --ablation --runs 5
```

Measure inference latency (Reviewer 1 & 2 requirement):
```bash
python main.py --mode balanced --measure_latency --runs 1
```

## 📊 CLI Arguments

- `--mode`: Reward profile to use
  - `balanced` (default): Balanced performance and stability
  - `high_stability`: Maximum stability - minimal switching
  - `vanilla`: Baseline: Pure SNR Maximization (Reviewer 2)
  - `high_coverage`: Maximum coverage - aggressive switching

- `--ablation`: Disable history features to test GRU importance (Reviewer 1 & 3)

- `--measure_latency`: Measure inference time per decision (Reviewer 1 & 2)

- `--gpu`: GPU ID (default: "0")

- `--runs`: Number of independent runs (default: 3)

## 🔬 CI/CD Workflows

The modular structure supports automated CI/CD pipelines. Example workflows:

### Generate Vanilla Baseline Data (Reviewer 2)
```bash
python main.py --mode vanilla --runs 5 > results_vanilla.txt
```

### Generate Ablation Study Data (Reviewer 1 & 3)
```bash
python main.py --mode balanced --ablation --runs 5 > results_ablation.txt
```

### Generate Latency Data (Reviewer 1 & 2)
```bash
python main.py --mode balanced --measure_latency --runs 1
```

### Generate Final Proposed Results
```bash
python main.py --mode balanced --runs 5 > results_proposed.txt
```

## 📄 Citation

If you use this code or find it useful, please cite:

```bibtex
@misc{natanzi2025onlinelearningbasedadaptivebeam,
  title={Online Learning-based Adaptive Beam Switching for 6G Networks: Enhancing Efficiency and Resilience}, 
  author={Seyed Bagher Hashemi Natanzi and Zhicong Zhu and Bo Tang},
  year={2025},
  eprint={2505.08032},
  archivePrefix={arXiv},
  primaryClass={cs.NI},
  url={https://arxiv.org/abs/2505.08032}, 
}
