# Online Learning Adaptive Beam Switching for 6G Networks

Official implementation of:

> **Online Learning-based Adaptive Beam Switching for 6G Networks: Enhancing Efficiency and Resilience**  
> *Seyed Bagher Hashemi Natanzi, Zhicong Zhu, Bo Tang*  
> [arXiv:2505.08032](https://arxiv.org/abs/2505.08032)

This repository implements an online learning-based adaptive beam switching framework for 6G networks. It features a Dueling DQN agent with Prioritized Experience Replay (PER) for real-time beam selection in dynamic 6G environments characterized by user mobility and time-correlated blockage.

## 🎯 Key Features

- **Dueling DQN Architecture**: Deep reinforcement learning agent with enhanced state representation
- **Prioritized Experience Replay**: Efficient learning from important experiences
- **NVIDIA Sionna Integration**: High-fidelity channel modeling using Rayleigh Block Fading
- **Multiple Reward Profiles**: Balanced, High-Stability, Vanilla, and High-Coverage modes
- **Ablation Study Support**: Evaluate the contribution of temporal features
- **Real-time Inference**: Sub-millisecond latency on H100 GPU

## 📁 Project Structure

```
OL-Beam-Switching-6G/
├── src/
│   ├── __init__.py
│   ├── config.py          # Constants, reward profiles, and hyperparameters
│   ├── models.py          # Dueling DQN neural network architecture
│   ├── core.py            # Prioritized replay buffer and state history
│   ├── environment.py     # Channel generation (Sionna), mobility, beamforming
│   └── baselines.py       # Greedy and MAB-UCB baseline algorithms
├── tests/                 # Unit tests
├── main.py                # Main entry point with CLI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: H100 or similar)
- NVIDIA drivers and CUDA toolkit

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CLIS-WPI/OL-Beam-Switching-6G.git
cd OL-Beam-Switching-6G
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup (Optional)

```bash
docker build -t ol-beam-switching .
docker run --gpus '"device=0"' --shm-size=8g -it --rm -v $(pwd):/workspace ol-beam-switching
```

## 💻 Usage

### Basic Usage

Run with default balanced mode (3 runs):
```bash
python main.py
```

### Command Line Arguments

- `--mode`: Reward profile selection
  - `balanced` (default): Balanced performance and stability
  - `high_stability`: Maximum stability with minimal switching
  - `vanilla`: Pure SNR maximization (no stability focus)
  - `high_coverage`: Maximum coverage with aggressive switching

- `--ablation`: Disable temporal features for ablation study
- `--measure_latency`: Measure inference latency per decision
- `--gpu`: GPU ID (default: "0")
- `--runs`: Number of independent runs (default: 3)

### Example Commands

**Vanilla baseline (pure SNR maximization):**
```bash
python main.py --mode vanilla --runs 5
```

**Ablation study (without temporal features):**
```bash
python main.py --mode balanced --ablation --runs 5
```

**Latency measurement:**
```bash
python main.py --mode balanced --measure_latency --runs 1
```

**Production run (balanced mode, 5 runs):**
```bash
python main.py --mode balanced --runs 5 > results_proposed.txt
```

## 📊 System Configuration

### Simulation Parameters

- **Users**: 100 mobile UEs
- **Antennas**: 64-element ULA at BS
- **Carrier Frequency**: 28 GHz
- **Transmit Power**: 38 dBm
- **Beams**: 64 DFT beams
- **Training Steps**: 20,000
- **Evaluation Steps**: 1,000

### DRL Hyperparameters

- **Architecture**: Dueling DQN (512-512-256 hidden units)
- **Replay Buffer**: 100,000 transitions
- **Batch Size**: 512
- **Learning Rate**: 3×10⁻⁴
- **Discount Factor**: 0.99
- **Target Update Frequency**: Every 100 steps

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 📈 Results

The framework achieves:
- **Stability Score**: 0.995 (lower is better, outperforms Vanilla DRL by 25%)
- **Average SNR**: 15.3 dB (matches MAB baseline)
- **Coverage Ratio**: 79.8%
- **Inference Latency**: 0.69 ms (on H100 GPU)

## 📄 Citation

If you use this code in your research, please cite:

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
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA Sionna for channel modeling capabilities
- PyTorch team for deep learning framework

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.
