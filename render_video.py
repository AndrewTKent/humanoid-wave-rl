import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Record video of trained humanoid')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_wave.mp4',
                      help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                      help='Number of frames to record')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode')
    return parser.parse_args()

def record_video_headless(model_path, output_path, num_frames=500, max_steps=1000):
    """Record video of the humanoid in a headless environment"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Recording video to {output_path}...")
    
    # Create environment with NO assistance for proper evaluation
    env = DMCWrapper(
        domain_name="humanoid",
        task_name="stand",
        initial_standing_assist=0.0,  # No assistance for evaluation
        max_steps=max_steps
    )
    
    # Reset with a specific seed for reproducibility
    obs, _ = env.reset(seed=42)
    
    frames = []
    total_reward = 0
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from model with some exploration noise for better performance
        action, _ = model.predict(obs, deterministic=False)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Get frame
        frame = env.render(mode='rgb_array', height=480, width=640, camera_id=0)
        frames.append(frame)
        
        # Print debug info
        if i % 100 == 0:
            print(f"  Current height: {info['height']:.2f}, Reward: {reward:.2f}")
        
        # Reset if done
        if done or truncated:
            print(f"Episode finished with total reward: {total_reward:.2f}")
            # Try a different seed each reset
            obs, _ = env.reset(seed=42+i)
            total_reward = 0
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    record_video_headless(
        args.model_path, 
        args.output_path, 
        args.num_frames,
        args.max_steps
    )

if __name__ == "__main__":
    main()
