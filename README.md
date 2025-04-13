# Stand - Humanoid RL

Train a humanoid to stand up using reinforcement learning with GPU acceleration.

## Project Overview

This project implements a reinforcement learning solution to teach a humanoid robot to stand up and maintain balance. The implementation leverages:

- **dm_control**: For the physics-based humanoid environment
- **stable-baselines3**: For the Proximal Policy Optimization (PPO) algorithm
- **Gymnasium**: For standardized environment interfaces
- **GPU Acceleration**: For faster training with parallel environments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/humanoid-stand-rl.git
cd humanoid-stand-rl

# Create a virtual environment
python3 -m venv stand

# Activate the virtual environment
source stand/bin/activate  # On Windows: stand\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Main script for training and evaluation
- `dmc_wrapper.py`: Wrapper to make dm_control environment compatible with Gymnasium
- `visualization.py`: Functions for model evaluation and video recording
- `requirements.txt`: Required dependencies

## Usage

### Training

```bash
# Train with default parameters (16 parallel environments)
python3 main.py

# Train with custom settings
python3 main.py --total_timesteps 500000 --num_envs 8
```

### Evaluation

```bash
# Evaluate and record video of the trained model
python3 main.py --mode evaluate --model_path results/humanoid_stand_final.zip
```

The script automatically renders and saves a video of the trained humanoid after both training and evaluation.

## Performance Optimization

The implementation includes several optimizations for better performance:

1. **Vectorized Environments**: Runs multiple environments in parallel to maximize throughput
2. **GPU Acceleration**: Uses CUDA for neural network computation when available
3. **Optimized Hyperparameters**: Batch size and network architecture tuned for GPU performance

## Approach

### Environment Wrapper

The project includes a custom wrapper that converts the dm_control humanoid environment to be compatible with stable-baselines3, including:
- Converting observations to a flat vector
- Handling step and reset functions
- Managing the reward function for standing behavior

### Training Parameters

The reinforcement learning approach uses PPO with:
- Deep neural network with [256, 256] hidden layers
- Learning rate of 3e-4
- Batch size optimized for CPU/GPU
- Gamma (discount factor) of 0.99

## Results

After training, the humanoid should be able to:
- Start from a lying down position
- Stand up
- Maintain balance in a standing position

The training progress and final results will be saved in the results directory, including:
- Checkpoint models during training
- Final trained model
- Video recording of the trained humanoid's behavior
