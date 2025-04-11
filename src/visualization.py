"""
Visualization utilities for humanoid wave RL.
"""

import os
import numpy as np
import imageio
from dm_control import viewer


def evaluate_model(env, model, num_episodes=3):
    """
    Evaluate a trained model on the environment.
    
    Args:
        env: Environment to evaluate on
        model: Trained model
        num_episodes: Number of episodes to evaluate
    """
    all_rewards = []
    all_stand_rewards = []
    all_wave_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        episode_stand_rewards = []
        episode_wave_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_stand_rewards.append(info.get('stand_reward', 0))
            episode_wave_rewards.append(info.get('wave_reward', 0))
        
        # Compute episode statistics
        total_reward = sum(episode_rewards)
        total_stand_reward = sum(episode_stand_rewards)
        total_wave_reward = sum(episode_wave_rewards)
        
        all_rewards.append(total_reward)
        all_stand_rewards.append(total_stand_reward)
        all_wave_rewards.append(total_wave_reward)
        
        print(f"Episode {episode+1}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Stand Reward: {total_stand_reward:.2f}")
        print(f"  Wave Reward: {total_wave_reward:.2f}")
    
    # Print overall statistics
    print("\nEvaluation Summary:")
    print(f"  Mean Total Reward: {np.mean(all_rewards):.2f}")
    print(f"  Mean Stand Reward: {np.mean(all_stand_rewards):.2f}")
    print(f"  Mean Wave Reward: {np.mean(all_wave_rewards):.2f}")


def record_video(env, model, video_path="humanoid_wave.mp4", num_frames=500):
    """
    Record a video of the humanoid controlled by the trained model without visualization.
    Works on headless servers by using OSMesa software rendering.
    
    Args:
        env: Environment to record from
        model: Trained model to control the humanoid
        video_path: Path to save the video
        num_frames: Number of frames to record
    """
    import os
    import numpy as np
    import imageio
    
    # Set environment variable for MuJoCo to use OSMesa
    os.environ['MUJOCO_GL'] = 'osmesa'
    
    print(f"Recording video to {video_path}...")
    
    # Create new environment for rendering
    # This avoids interfering with the original environment
    render_env = env.env
    
    # Set up camera
    camera_id = 0  # Use the default free camera
    height = 480
    width = 640
    
    # Initialize
    frames = []
    timestep = render_env.reset()
    
    # Record frames
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get observation and predict action using model
        obs = env._flatten_obs(timestep.observation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        timestep = render_env.step(action)
        
        # Render frame
        pixels = render_env.physics.render(height=height, width=width, camera_id=camera_id)
        frames.append(pixels)
        
        # Check if episode is done
        if timestep.last():
            timestep = render_env.reset()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(video_path)) or '.', exist_ok=True)
    
    # Save frames as video
    print(f"Saving video to {video_path}...")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved successfully!")
