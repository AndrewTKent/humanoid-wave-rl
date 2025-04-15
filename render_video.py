import os
# Force MuJoCo to use EGL for rendering
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import numpy as np
import imageio
import wandb
from stable_baselines3 import PPO
from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Record video of trained humanoid')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_video.mp4',
                      help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                      help='Number of frames to record')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Upload video to wandb')
    return parser.parse_args()

def record_video_headless(model_path, output_path, num_frames=500, max_steps=1000, use_wandb=False):
    """Record video of the humanoid in a headless environment"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Recording video to {output_path}...")
    
    # Create environment with NO assistance for proper evaluation
    env = DMCWrapper(
        domain_name="humanoid",
        task_name="stand",
        max_steps=max_steps
    )
    
    # Reset with a specific seed for reproducibility
    obs, _ = env.reset(seed=42)
    
    frames = []
    total_reward = 0
    episode_rewards = []
    episode_heights = []
    
    # Extract model name for the wandb run
    model_name = os.path.basename(os.path.dirname(model_path))
    timesteps = os.path.basename(output_path).replace('humanoid_video_', '').replace('.mp4', '')
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Track episode metrics
        if 'height' in info:
            episode_heights.append(info['height'])
        
        try:
            # Get frame with explicit error handling
            frame = env.render(mode='rgb_array', height=480, width=640, camera_id=0)
            frames.append(frame)
        except Exception as e:
            print(f"Error rendering frame {i+1}: {e}")
            break
        
        # Print debug info
        if i % 100 == 0:
            print(f"  Current height: {info['height']:.2f}, Reward: {reward:.2f}")
        
        # Reset if done
        if done or truncated:
            print(f"Episode finished with total reward: {total_reward:.2f}")
            episode_rewards.append(total_reward)
            # Try a different seed each reset
            obs, _ = env.reset(seed=42+i)
            total_reward = 0
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if frames:
        # Save the video
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved to {output_path}")
        
        # Upload to wandb if requested
        if use_wandb:
            # Initialize wandb if not already initialized
            if not wandb.run:
                wandb.init(project="humanoid-stand", name=f"video-{model_name}-{timesteps}")
            
            # Calculate evaluation metrics
            eval_results = {
                "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
                "mean_height": np.mean(episode_heights) if episode_heights else 0,
                "max_height": np.max(episode_heights) if episode_heights else 0,
            }
            
            # Log video and metrics
            wandb.log({
                "video": wandb.Video(output_path, fps=30, format="mp4"),
                "final_eval/mean_reward": eval_results["mean_reward"],
                "final_eval/mean_height": eval_results["mean_height"],
                "final_eval/max_height": eval_results["max_height"],
            })
            
            print(f"Video uploaded to wandb")
    else:
        print("No frames were rendered, cannot save video")

def main():
    args = parse_args()
    record_video_headless(
        args.model_path, 
        args.output_path, 
        args.num_frames,
        args.max_steps,
        args.use_wandb
    )

if __name__ == "__main__":
    main()
