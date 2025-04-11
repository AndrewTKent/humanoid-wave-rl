# Stand and Wave

Train a humanoid to stand up and wave using reinforcement learning.

## Project Overview

This project implements a reinforcement learning solution to teach a humanoid robot two sequential tasks:
1. Stand up and maintain balance
2. Wave with one arm while maintaining a standing position

The implementation leverages:
- **dm_control** for the physics-based humanoid environment
- **stable-baselines3** for the Proximal Policy Optimization (PPO) algorithm
- **Gymnasium** as the standardized environment interface

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewtkent/humanoid-wave-rl.git
cd humanoid-wave-rl

# Create a virtual environment called 'wave'
python -m venv wave

# Activate the virtual environment
source wave/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default parameters
python main.py
```

### Evaluation

```bash
# Evaluate and record video of the trained model
python main.py --model_path results/humanoid_wave_final.zip --mode evaluate
```

## Approach

### Environment Wrapper

The project includes a custom wrapper that converts the dm_control humanoid environment to be compatible with stable-baselines3, including:
- Converting observations to a flat vector
- Handling step and reset functions
- Implementing the reward shaping for waving behavior

### Reward Design

The reward function combines:

1. **Standing Reward**: The original reward from dm_control
2. **Wave Reward**: A custom reward that encourages arm movement
   - Tracks specific arm joint positions
   - Rewards oscillatory movement when the arm is elevated
   - Uses curriculum learning to balance standing and waving

## Design Decisions

1. **Right Arm Joint Selection**: Joints 5-7 were selected based on experimental observation of the humanoid model. These appear to control shoulder and elbow movement.

2. **Curriculum Learning**: The waving reward is gradually introduced after the agent learns to stand, using a progress factor that increases over time.

3. **Wave Definition**: A "wave" is defined as oscillatory movement of the arm while it's held in an elevated position.

## Results

After training for approximately 1 million timesteps, the agent successfully:
- Stands up and maintains balance
- Raises its right arm
- Performs a consistent waving motion
