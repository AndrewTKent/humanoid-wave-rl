# Stand and Wave

Train a humanoid to stand up and wave using reinforcement learning with GPU acceleration.

## Project Overview

This project implements a reinforcement learning solution to teach a humanoid robot two sequential tasks:
1. Stand up and maintain balance
2. Wave with one arm while maintaining a standing position

The implementation leverages:
- **dm_control**: For the physics-based humanoid environment
- **stable-baselines3**: For the Proximal Policy Optimization (PPO) algorithm
- **Gymnasium**: For standardized environment interfaces
- **GPU Acceleration**: For faster training with parallel environments

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewtkent/humanoid-wave-rl.git
cd humanoid-wave-rl

# Create a virtual environment called 'wave'
python -m venv wave

# Activate the virtual environment
source wave/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default parameters (16 parallel environments)
python main.py

# Train with custom settings
xvfb-run -a python main.py --total_timesteps 3000000 --num_envs 24 --wandb --device cuda
```

### Evaluation

```bash
# Evaluate and record video of the trained model
python main.py --mode evaluate --model_path results/humanoid_wave_final.zip
```

### Video

For headless systems (which is the recommended approach for servers without a display):

```bash
# Generate and save video of the trained model using xvfb for headless rendering
xvfb-run -a python render_video.py --model_path results/humanoid_stand_final.zip --output_path output/humanoid_video.mp4
```

**Note**: If you encounter the following error:
```
ValueError: Could not find a backend to open `output/humanoid_video.mp4`` with iomode `wI`.
Based on the extension, the following plugins might add capable backends:
  FFMPEG:  pip install imageio[ffmpeg]
  pyav:  pip install imageio[pyav]
```

Install the required backend with:
```bash
pip install imageio[ffmpeg]
```

If you have a display available, you can run without xvfb, but this is not recommended for most server environments:
```bash
python render_video.py --model_path results/humanoid_stand_final.zip --output_path output/humanoid_video.mp4
```

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

4. **Parallel Training**: Using multiple environments in parallel significantly speeds up training.

## Results

After training for approximately 1 million timesteps, the agent successfully:
- Stands up and maintains balance
- Raises its right arm
- Performs a consistent waving motion
