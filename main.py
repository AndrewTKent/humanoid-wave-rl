"""
Main script for humanoid wave training and evaluation with optimized performance.
"""

import os
import time
import argparse
import torch
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.dmc_wrapper import DMCWrapper
from src.visualization import evaluate_model, record_video, record_closeup_video_headless


class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress as percentage and estimated time.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -0.1  # Initialize to -0.1 to print at 0.0%
        self.start_time = time.time()
    
    def _on_step(self):
        """Called after each step of the environment"""
        # Calculate percentage with one decimal place
        percent = round(100 * self.num_timesteps / self.total_timesteps, 1)
        
        # Only print when percentage changes by at least 0.1%
        if percent > self.last_percent + 0.009:  # Use 0.09 to account for float precision
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - self.start_time
            if self.num_timesteps > 0:
                time_per_step = elapsed_time / self.num_timesteps
                remaining_steps = self.total_timesteps - self.num_timesteps
                remaining_time = remaining_steps * time_per_step
                
                # Format times as hours:minutes:seconds
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            else:
                # Avoid division by zero at first step
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Just started")
            
            self.last_percent = percent
        
        return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Humanoid Wave Training')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'], 
                       help='Mode: train or evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (for evaluation)')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total timesteps for training')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--num_envs', type=int, default=16,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run on (auto, cpu, or cuda)')
    
    return parser.parse_args()


def make_env():
    """Create a function that returns a DMCWrapper environment."""
    def _init():
        return DMCWrapper()
    return _init


def train_humanoid_wave(total_timesteps=1000000, output_dir='results', num_envs=16, device='auto'):
    """Train the humanoid to stand and wave with parallel environments."""
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            print("CUDA is available, but training may be faster on CPU for MlpPolicy.")
            print("You can force CPU usage with --device cpu or continue with GPU.")
            device = "cuda"  # Still use CUDA by default
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
    
    # Create vectorized environment with multiple parallel instances
    print(f"Creating {num_envs} parallel environments...")
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1), 
        save_path=os.path.join(output_dir, f"checkpoints_{timestamp}"),
        name_prefix="humanoid_wave"
    )
    
    # Set up progress callback
    progress_callback = ProgressCallback(total_timesteps)
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64 if device == "cpu" else 256,  # Smaller for CPU, larger for GPU
        n_epochs=10,
        gamma=0.99,
        device=device,
        policy_kwargs={"net_arch": [256, 256]}  # Deep network architecture
    )
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    print(f"Progress will be shown as iterations, where each iteration processes")
    print(f"{num_envs * model.n_steps} timesteps ({num_envs} envs * {model.n_steps} steps)")
    
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_callback])
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "humanoid_wave_final.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # For evaluation, we need a non-vectorized environment
    eval_env = DMCWrapper()
    
    # Evaluate the model
    evaluate_model(eval_env, model)
    
    # Record videos
    print("Recording videos of the trained humanoid...")
    video_path = os.path.join(output_dir, f"humanoid_wave_{timestamp}.mp4")
    record_video(eval_env, model, video_path)
    
    # Record a close-up of the waving motion
    closeup_path = os.path.join(output_dir, f"humanoid_wave_closeup_{timestamp}.mp4")
    record_closeup_video_headless(eval_env, model, closeup_path)
    
    print(f"Training and evaluation complete.")
    print(f"Full video: {video_path}")
    print(f"Close-up video: {closeup_path}")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        # Train mode
        train_humanoid_wave(
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            num_envs=args.num_envs,
            device=args.device
        )
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        
        # Load model
        model = PPO.load(args.model_path)
        
        # Create environment
        env = DMCWrapper()
        
        # Evaluate
        evaluate_model(env, model)
        
        # Record videos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(args.output_dir, f"humanoid_wave_{timestamp}.mp4")
        record_video(env, model, video_path)
        
        # Record a close-up of the waving motion
        closeup_path = os.path.join(args.output_dir, f"humanoid_wave_closeup_{timestamp}.mp4")
        record_waving_closeup(env, model, closeup_path)
        
        print(f"Evaluation complete.")
        print(f"Full video: {video_path}")
        print(f"Close-up video: {closeup_path}")


if __name__ == "__main__":
    main()
