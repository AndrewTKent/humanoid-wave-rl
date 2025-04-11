import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
from dm_control import suite
from dm_control.suite.wrappers import pixels

from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Record video of trained humanoid')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_wave.mp4',
                       help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                       help='Number of frames to record')
    return parser.parse_args()

def record_video_headless(model_path, output_path, num_frames=500):
    """Record video of the humanoid in a headless environment"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Recording video to {output_path}...")
    
    # Regular environment for action prediction
    env = DMCWrapper()
    
    # Environment with rendering capabilities
    render_env = suite.load(domain_name="humanoid", task_name="stand")
    render_env = pixels.Wrapper(render_env)
    
    # Reset environments
    obs, _ = env.reset()
    time_step = render_env.reset()
    
    frames = []
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environments
        obs, reward, done, _, _ = env.step(action)
        time_step = render_env.step(action)
        
        # Get frame
        frame = render_env.physics.render(height=480, width=640, camera_id=0)
        frames.append(frame)
        
        # Reset if done
        if done:
            obs, _ = env.reset()
            time_step = render_env.reset()
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    record_video_headless(args.model_path, args.output_path, args.num_frames)

if __name__ == "__main__":
    main()
