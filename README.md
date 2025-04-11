# Humanoid Wave RL

Train a humanoid robot to stand up and wave using reinforcement learning.

![Humanoid Waving Demo](assets/humanoid_wave.gif)

## Project Overview

This project trains a humanoid robot to perform two sequential tasks:
1. Stand up and maintain balance
2. Wave with one arm while maintaining standing position

The implementation uses:
- **dm_control**: For the physics-based humanoid environment
- **stable-baselines3**: For the PPO (Proximal Policy Optimization) implementation
- **Gymnasium**: For standardized environment interfaces

## Key Features

- Custom wrapper to convert dm_control environments to Gymnasium format
- Reward shaping to encourage both standing and waving behavior
- Curriculum learning approach that prioritizes standing before waving
- Visualization tools for monitoring training progress and recording videos

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/humanoid-wave-rl.git
cd humanoid-wave-rl

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default parameters
python main.py

# Or run the training script directly with custom parameters
python scripts/train.py --timesteps 2000000 --learning_rate 3e-4
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model_path results/model.zip

# Record a video of the trained agent
python scripts/record_video.py --model_path results/model.zip
```

## Approach & Design Decisions

### Environment Wrapper

The project includes a custom wrapper that converts the dm_control humanoid environment to be compatible with stable-baselines3. The wrapper:

- Flattens the dictionary-based observation space into a single vector
- Defines appropriate Gymnasium spaces for observations and actions
- Implements a custom reward function for the waving behavior

### Reward Design

The reward function combines:

1. **Standing Reward**: The original reward from the dm_control environment
2. **Wave Reward**: A custom reward that detects oscillatory arm movements
   - Tracks shoulder joint position and movement
   - Rewards direction changes when the arm is elevated
   - Provides continuous reward for maintaining the arm in an elevated position

### Curriculum Learning

To manage the complexity of learning both tasks:

- Initially prioritizes the standing reward
- Gradually introduces the waving reward component
- Uses a progress factor to balance rewards over time

## Results

After training for 1 million timesteps, the agent successfully:
- Stands up and maintains balance
- Raises its right arm
- Performs a waving motion with the raised arm

Training metrics show:
- Standing reward stabilized after approximately 300K timesteps
- Waving behavior emerged around 500K timesteps
- Combined performance plateaued around 800K timesteps

## Future Improvements

- Experiment with different arm joints for more natural waving
- Implement different waving styles (side-to-side vs up-down)
- Try different RL algorithms beyond PPO
- Add more complex behaviors like walking and waving simultaneously
- Implement multi-task learning for a variety of gestures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DeepMind Control Suite for the humanoid environment
- The stable-baselines3 team for their implementation of PPO
