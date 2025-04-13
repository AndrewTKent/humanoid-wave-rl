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
    """
    all_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_rewards.append(reward)
        
        # Compute episode statistics
        total_reward = sum(episode_rewards)
        all_rewards.append(total_reward)
        
        print(f"Episode {episode+1}:")
        print(f"  Total Reward: {total_reward:.2f}")
    
    # Print overall statistics
    print("\nEvaluation Summary:")
    print(f"  Mean Total Reward: {np.mean(all_rewards):.2f}")


def record_video(env, model, video_path="humanoid_stand.mp4", num_frames=500):
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
