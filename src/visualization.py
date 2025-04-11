"""
Visualization utilities for the humanoid environment.

This module provides functions for visualizing the humanoid
and recording videos of its behavior.
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
        num_episodes (int): Number of episodes to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    all_rewards = []
    all_stand_rewards = []
    all_wave_rewards = []
    all_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        episode_stand_rewards = []
        episode_wave_rewards = []
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_stand_rewards.append(info.get('stand_reward', 0))
            episode_wave_rewards.append(info.get('wave_reward', 0))
            step_count += 1
        
        # Compute episode statistics
        total_reward = sum(episode_rewards)
        total_stand_reward = sum(episode_stand_rewards)
        total_wave_reward = sum(episode_wave_rewards)
        
        all_rewards.append(total_reward)
        all_stand_rewards.append(total_stand_reward)
        all_wave_rewards.append(total_wave_reward)
        all_lengths.append(step_count)
        
        print(f"Episode {episode+1}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Stand Reward: {total_stand_reward:.2f}")
        print(f"  Wave Reward: {total_wave_reward:.2f}")
        print(f"  Length: {step_count}")
    
    # Compute overall statistics
    metrics = {
        'mean_reward': np.mean(all_rewards),
        'mean_stand_reward': np.mean(all_stand_rewards),
        'mean_wave_reward': np.mean(all_wave_rewards),
        'mean_length': np.mean(all_lengths),
        'std_reward': np.std(all_rewards),
        'max_reward': np.max(all_rewards),
        'min_reward': np.min(all_rewards),
    }
    
    print("\nEvaluation Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    return metrics


def record_video(env, model, video_path="humanoid_wave.mp4", num_frames=500, height=480, width=640, fps=30):
    """
    Record a video of the humanoid controlled by the trained model.
    
    Args:
        env: Environment to record from
        model: Trained model to control the humanoid
        video_path (str): Path to save the video
        num_frames (int): Number of frames to record
        height (int): Video height
        width (int): Video width
        fps (int): Frames per second
        
    Returns:
        str: Path to the saved video
    """
    print(f"Recording video with {num_frames} frames...")
    
    # Create a policy function that uses the trained model
    def policy_fn(time_step):
        obs = env._flatten_obs(time_step.observation)
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Use dm_control's viewer to render frames
    frames = []
    with viewer.launch_passive(env.env, policy=policy_fn) as viewer_instance:
        # Record frames
        for i in range(num_frames):
            if i % 50 == 0:
                print(f"  Rendering frame {i+1}/{num_frames}")
            
            viewer_instance.step()
            pixels = viewer_instance.render(height=height, width=width)
            frames.append(pixels)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    
    # Save frames as video
    print(f"Saving video to {video_path}...")
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved successfully!")
    
    return video_path


def plot_training_progress(log_file, output_path=None):
    """
    Plot training progress from a log file.
    
    Args:
        log_file (str): Path to the log file
        output_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load log data
    df = pd.read_csv(log_file)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot total reward
    axes[0, 0].plot(df['timestep'], df['reward'], 'b-')
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot stand reward
    axes[0, 1].plot(df['timestep'], df['stand_reward'], 'g-')
    axes[0, 1].set_title('Stand Reward')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # Plot wave reward
    axes[1, 0].plot(df['timestep'], df['wave_reward'], 'r-')
    axes[1, 0].set_title('Wave Reward')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True)
    
    # Plot episode length
    axes[1, 1].plot(df['timestep'], df['episode_length'], 'k-')
    axes[1, 1].set_title('Episode Length')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Length')
    axes[1, 1].grid(True)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    return fig
