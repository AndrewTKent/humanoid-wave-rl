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


def record_waving_closeup(env, model, video_path="humanoid_wave_closeup.mp4", num_frames=300):
    """
    Record a video focused specifically on the waving motion.
    
    This creates a closer view of the upper body to highlight the waving.
    
    Args:
        env: Environment to record from
        model: Trained model to control the humanoid
        video_path: Path to save the video
        num_frames: Number of frames to record
    """
    print(f"Recording close-up waving video to {video_path}...")
    
    # Create a policy function that uses the trained model
    def policy_fn(time_step):
        obs = env._flatten_obs(time_step.observation)
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Use dm_control's viewer to render frames
    frames = []
    
    # Configure the camera for upper body close-up view
    camera_settings = {
        'distance': 3.0,       # Closer to the subject
        'azimuth': 30.0,       # From the side to see arm movement
        'elevation': -5.0,     # Slightly from above
        'lookat': [0, 0, 1.2]  # Focus on upper body
    }
    
    with viewer.launch_passive(env.env, policy=policy_fn) as viewer_instance:
        # Set up the camera
        viewer_instance.camera.set_params(**camera_settings)
        
        # Warm-up period - let the humanoid stabilize before recording
        for _ in range(100):
            viewer_instance.step()
        
        # Record frames
        for i in range(num_frames):
            if i % 50 == 0:
                print(f"  Rendering frame {i+1}/{num_frames}")
            
            viewer_instance.step()
            
            # HD resolution for better quality
            pixels = viewer_instance.render(height=720, width=1280, camera_id=0)
            frames.append(pixels)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    
    # Save frames as video with higher quality
    print(f"Saving video to {video_path}...")
    imageio.mimsave(video_path, frames, fps=30, quality=8, macro_block_size=16)
    print(f"Video saved successfully!")
    
    # Return the path to the saved video
    return video_path
