"""
Visualization utilities for humanoid standing RL.
"""

import os
import numpy as np
import imageio
from dm_control import suite
from dm_control.suite.wrappers import pixels


def evaluate_model(env, model, num_episodes=3):
    """
    Evaluate a trained model on the environment.
    
    Args:
        env: Environment to evaluate on
        model: Trained model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation statistics
    """
    all_rewards = []
    all_episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        steps = 0
        
        while not done:
            steps += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_rewards.append(reward)
        
        # Compute episode statistics
        total_reward = sum(episode_rewards)
        all_rewards.append(total_reward)
        all_episode_lengths.append(steps)
        
        print(f"Episode {episode+1}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Episode Length: {steps} steps")
    
    # Print overall statistics
    mean_reward = np.mean(all_rewards)
    mean_length = np.mean(all_episode_lengths)
    
    print("\nEvaluation Summary:")
    print(f"  Mean Total Reward: {mean_reward:.2f}")
    print(f"  Mean Episode Length: {mean_length:.2f} steps")
    print(f"  Min/Max Reward: {min(all_rewards):.2f}/{max(all_rewards):.2f}")
    
    # Return statistics for logging
    return {
        "mean_reward": mean_reward,
        "mean_episode_length": mean_length,
        "min_reward": min(all_rewards),
        "max_reward": max(all_rewards),
        "std_reward": np.std(all_rewards),
        "all_rewards": all_rewards,
        "all_episode_lengths": all_episode_lengths
    }


def record_video(env, model, video_path="humanoid_stand.mp4", num_frames=500):
    """
    Record a video of the humanoid in a headless environment.
    
    Args:
        env: DMCWrapper environment
        model: Trained model
        video_path: Path to save the video
        num_frames: Number of frames to record
        
    Returns:
        Path to the saved video
    """
    print(f"Recording video to {video_path}...")
    
    # Create environment with pixel rendering enabled
    env_with_pixels = suite.load(domain_name="humanoid", task_name="stand")
    env_with_pixels = pixels.Wrapper(env_with_pixels, render_kwargs={'height': 480, 'width': 640})
    
    # Reset environment
    time_step = env_with_pixels.reset()
    frames = []
    
    # Track metrics during recording
    total_reward = 0
    min_reward = float('inf')
    max_reward = float('-inf')
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Convert observation for policy
        obs = env._flatten_obs(time_step.observation)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        time_step = env_with_pixels.step(action)
        
        # Track reward
        if time_step.reward is not None:
            reward = float(time_step.reward)
            total_reward += reward
            min_reward = min(min_reward, reward)
            max_reward = max(max_reward, reward)
        
        # Render and store frame
        pixels = env_with_pixels.physics.render(height=480, width=640, camera_id=0)
        frames.append(pixels)
        
        # Check if episode is done
        if time_step.last():
            time_step = env_with_pixels.reset()
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")
    
    # Print statistics from the recording
    if min_reward != float('inf'):
        print(f"  Recording stats - Total reward: {total_reward:.2f}, Min/Max reward: {min_reward:.2f}/{max_reward:.2f}")
    
    return video_path
