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
        env: The environment to evaluate on
        model: The trained model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Initialize metrics
    total_rewards = []
    stand_rewards = []
    wave_rewards = []
    episode_lengths = []
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        stand_reward = 0
        wave_reward = 0
        steps = 0
        
        # Run one episode
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Accumulate rewards
            total_reward += reward
            stand_reward += info.get('stand_reward_step', 0)
            wave_reward += info.get('wave_reward_step', 0)
            steps += 1
        
        # Store episode results
        total_rewards.append(total_reward)
        stand_rewards.append(stand_reward)
        wave_rewards.append(wave_reward)
        episode_lengths.append(steps)
        
        # Print episode results
        print(f"Episode {episode}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Stand Reward: {stand_reward:.2f}")
        print(f"  Wave Reward: {wave_reward:.2f}")
    
    # Calculate summary metrics
    mean_total_reward = sum(total_rewards) / len(total_rewards)
    mean_stand_reward = sum(stand_rewards) / len(stand_rewards)
    mean_wave_reward = sum(wave_rewards) / len(wave_rewards)
    mean_episode_length = sum(episode_lengths) / len(episode_lengths)
    
    # Print summary
    print(f"Evaluation Summary:")
    print(f"  Mean Total Reward: {mean_total_reward:.2f}")
    print(f"  Mean Stand Reward: {mean_stand_reward:.2f}")
    print(f"  Mean Wave Reward: {mean_wave_reward:.2f}")
    
    # Return results dictionary
    results = {
        "mean_total_reward": mean_total_reward,
        "mean_stand_reward": mean_stand_reward,
        "mean_wave_reward": mean_wave_reward,
        "mean_episode_length": mean_episode_length,
        "total_rewards": total_rewards,
        "stand_rewards": stand_rewards,
        "wave_rewards": wave_rewards,
        "episode_lengths": episode_lengths
    }
    
    return results


def record_video(env, model, video_path="humanoid_wave.mp4", num_frames=500):
    """
    Record a video of the humanoid in a headless environment.
    """
    print(f"Recording video to {video_path}...")
    
    # Create an environment for recording
    import numpy as np
    import imageio
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

    # Create environment with pixel rendering enabled
    env_with_pixels = suite.load(domain_name="humanoid", task_name="stand")
    env_with_pixels = pixels.Wrapper(env_with_pixels, render_kwargs={'height': 480, 'width': 640})
    
    # Reset environment
    time_step = env_with_pixels.reset()
    frames = []
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Convert observation for policy
        obs = env._flatten_obs(time_step.observation)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        time_step = env_with_pixels.step(action)
        
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
    
    return video_path


def record_closeup_video_headless(model_path, output_path, num_frames=500):
    """Record a close-up video focused on the waving action"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Recording close-up video to {output_path}...")
    
    # Regular environment for action prediction
    env = DMCWrapper()
    
    # Environment with rendering capabilities
    render_env = suite.load(domain_name="humanoid", task_name="stand")
    render_env = pixels.Wrapper(render_env)
    
    # Reset environments
    obs, _ = env.reset()
    time_step = render_env.reset()
    
    frames = []
    
    # Let the humanoid stabilize before recording
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        time_step = render_env.step(action)
        if done:
            obs, _ = env.reset()
            time_step = render_env.reset()
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environments
        obs, reward, done, _, _ = env.step(action)
        time_step = render_env.step(action)
        
        # Get frame with camera focused on upper body
        # The camera_id=1 often gives a closer view
        frame = render_env.physics.render(
            height=720, 
            width=1280, 
            camera_id=1
        )
        frames.append(frame)
        
        # Reset if done
        if done:
            obs, _ = env.reset()
            time_step = render_env.reset()
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")
