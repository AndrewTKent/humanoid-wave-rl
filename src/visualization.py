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
