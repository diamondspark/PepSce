# ACP Generative Modeling with Reinforcement Learning

This repository implements a generative modeling framework for designing antimicrobial peptides (ACPs) using Evolutionary Scale Modeling (ESM) embeddings and reinforcement learning (RL) with Soft Actor-Critic (SAC). The project leverages protein language models to generate and optimize peptide sequences in a navigation-based environment.

## Features

- **ESM Integration**: Uses ESM-1b for extracting embeddings from peptide sequences.
- **Reinforcement Learning**: Employs SAC for sequence optimization in a custom navigation environment.
- **Parallel Processing**: Supports multiprocessing for batch embedding extraction during screening.
- **Evaluation**: Includes tools for policy evaluation and sampling new sequences.
- **Configurable**: Highly configurable via YAML files for experiments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/diamonspark/PepSce.git
   cd PepSce
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install ESM:
   ```bash
   pip install git+https://github.com/facebookresearch/esm.git
   ```

4. Additional requirements for RL and data processing:
   - stable-baselines3
   - torch
   - numpy
   - pandas
   - scikit-learn
   - tqdm
   - easydict
   - pyyaml
   - wandb (optional, for logging)

## Usage

### Training the Model

Run the training script to train the SAC policy:

```bash
python src/Driver.py
```

This script performs the following steps:
- Loads configuration from `params.yml`
- Initializes the navigation environment with training embeddings
- Creates a SAC agent with a multi-layer perceptron policy
- Trains the model for the specified number of timesteps
- Saves model checkpoints every 5000 timesteps
- Optionally processes sampling FASTA files in parallel using multiprocessing

**Key Configuration Parameters:**
- `rl.total_timesteps`: Total number of training steps
- `rl.model_savedir`: Directory to save model checkpoints
- `rl.model`: RL algorithm (SAC, PPO, DDPG, etc.)
- `tarsa_screening`: Enable/disable parallel screening after training

### Sampling and Evaluation

After training, use the trained policy for sampling and evaluation:

```bash
python src/screening/sample_mu.py
```

This script:
- Loads the trained SAC model from disk
- Initializes the sampling environment with pre-computed navigation data
- Evaluates the policy over 100,000,000 episodes to gather statistics
- Computes mean and standard deviation of rewards
- Maps peptide sequences to reward predictions

## Configuration

The project uses a YAML configuration file (`params.yml`) to manage hyperparameters. Key sections include:

**RL Configuration:**
```yaml
rl:
  model: SAC                          # Algorithm (SAC, PPO, etc.)
  total_timesteps: 2000000            # Total training steps
  model_savedir: ./models/tarsa/      # Checkpoint save directory
  train_batch: 10000                  # Number of training sequences
  samp_batch: 2000000                 # Number of sampling sequences
```

**Screening Configuration:**
```yaml
screening:
  num_workers: 8                      # Number of parallel workers
  policy_load_path: ./models/tarsa/policy.zip  # Trained model path
```

**Data Paths:**
```yaml
pca_load_path: ./models/pca_model.pkl          # Pre-trained PCA model
```

## Data

### Input Data
- `data/peptides.fasta`: Master FASTA file containing all peptide sequences

### Generated Data
- `data/emb_esm1l6/`: ESM embeddings for training sequences
- `data/samp_*.fasta`: Sampling batches created by FASTA splitting
- `data/emb_esm1l6_samp_*/`: ESM embeddings for each sampling batch
- `data/train_nav_data.pkl`: Precomputed PCA-reduced training navigation data
- `data/samp*_nav_data.pkl`: Precomputed PCA-reduced sampling navigation data

### Models
- `models/train_pcamodel.pkl`: Fitted PCA model for dimensionality reduction
- `models/tarsa/checkpoints/`: SAC model checkpoints saved during training

**Setup Instructions:**
1. Ensure `data/peptides.fasta` is present in the data directory
2. Configure paths in `params.yml` to match your environment
3. Pre-compute PCA model or let the script generate it automatically

## Project Structure

```
.
├── params.yml                        # Configuration file
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── data/
│   ├── peptides.fasta                # Input peptide sequences
│   ├── emb_esm1l6/                   # Training ESM embeddings
│   ├── samp_*.fasta                  # Sampling batch FASTA files
│   ├── train_nav_data.pkl            # Training navigation data
│   └── samp*_nav_data.pkl            # Sampling navigation data
├── models/
│   ├── train_pcamodel.pkl            # Trained PCA model
│   └── tarsa/
│       ├── checkpoints/              # Model checkpoints
│       └── *.pkl                     # Goal-to-potency mappings
├── src/
│   ├── Driver.py                     # Main training script
│   ├── screening/
│   │   └── sample_mu.py              # Sampling and evaluation script
│   ├── dataset/
│   │   ├── esm.py                    # ESM embedding extraction
│   │   ├── navData.py                # Navigation data handling
│   │   └── utils.py                  # Data utilities
│   ├── envs/
│   │   ├── NavEnv.py                 # Training RL environment
│   │   └── sampEnv.py                # Sampling RL environment
│   └── utils.py                      # General utilities (FASTA splitting)
└── logs/                             # TensorBoard logs and outputs
```

## Key Components

### Main Scripts

**Driver.py** - Training orchestration
- Loads ESM embeddings and navigation data
- Initializes the BuildingEnv environment
- Trains SAC agent with checkpoint callbacks
- Optionally runs parallel ESM extraction for sampling data

**sample_mu.py** - Inference and evaluation
- Loads pre-trained SAC policy
- Evaluates on sampling environment
- Computes reward statistics for peptide designs

### Core Modules

**NavEnv.py (BuildingEnv)**
- Custom Gym environment for peptide optimization
- State: 2D PCA coordinates (pca_x, pca_y)
- Action: continuous movement in 2D space
- Reward: function of goal-to-potency mapping

**sampEnv.py (SamplingEnv)**
- Variant of NavEnv for evaluation on sampling data
- Uses pre-computed goal-to-potency dictionary
- Enables policy performance assessment

**navData.py (NavData)**
- Handles ESM embedding loading
- Performs PCA dimensionality reduction
- Manages precomputed navigation data caching
- Supports both training and sampling data paths

**esm.py**
- Wrapper for ESM model inference
- Extracts protein embeddings from sequences
- Supports batch processing of FASTA files

## Workflow

1. **Preparation Phase**
   - Place `peptides.fasta` in `data/` directory
   - Configure `params.yml` with desired hyperparameters

2. **Training Phase** (run `src/Driver.py`)
   - Split FASTA into training and sampling batches
   - Extract ESM embeddings for training data
   - Apply PCA for dimensionality reduction
   - Train SAC agent in navigation environment
   - Save checkpoints every 5000 timesteps

3. **Screening Phase** (if enabled in config)
   - Extract ESM embeddings for sampling batches in parallel
   - Create navigation data for sampling sets

4. **Evaluation Phase** (run `src/screening/sample_mu.py`)
   - Load trained policy
   - Evaluate on sampling environment
   - Generate reward predictions for peptides

## Hardware Requirements

- **GPU**: Recommended for training (CUDA-enabled device)
- **Memory**: 16+ GB RAM for large embedding datasets
- **Storage**: 50+ GB for embeddings and checkpoints depending on dataset size

## Troubleshooting

**CUDA Not Available**
- Ensure PyTorch is installed with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**OOM Errors During Training**
- Reduce `buffer_size` in SAC model initialization
- Decrease batch size in NavData loading
- Use gradient checkpointing in ESM extraction

**Path Issues**
- Update absolute paths in `params.yml` to match your system
- Ensure all data directories exist before running scripts

**Multiprocessing Errors**
- On macOS, the script sets `spawn` method which may conflict with Jupyter
- Run `Driver.py` as a standalone script, not from notebooks

**ESM Model Download**
- First run downloads ESM model (~300MB) - ensure internet connection
- Models cached in `~/.cache/torch/hub/checkpoints/`

## License

This project is licensed under the MIT License.

## Citation

If you use this work, please cite:
```bibtex
@software{pepscegeneration2024,
  title={ACP Generative Modeling with Reinforcement Learning},
  author={Pandey, M.K.},
  year={2024}
}
```