"""
Main script for humanoid stand training and evaluation with wandb tracking.
"""

import os
import time
import argparse
import torch
from datetime import datetime
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.dmc_wrapper import DMCWrapper
from src.visualization import evaluate_model, record_video, record_closeup_video_headless


class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress and logging to wandb.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -0.1
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self):
        """Called after each step of the environment"""
        # Accumulate rewards and lengths
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            # Log to wandb
            wandb.log({
                "episode_reward": self.current_episode_reward,
                "episode_length": self.current_episode_length,
                "stand_reward": self.locals['infos'][0].get('stand_reward', 0),
                "wave_reward": self.locals['infos'][0].get('wave_reward', 0),
            })
            # Reset episode accumulators
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Calculate and log progress
        percent = round(100 * self.num_timesteps / self.total_timesteps, 1)
        if percent > self.last_percent + 0.09:
            elapsed_time = time.time() - self.start_time
            if self.num_timesteps > 0:
                time_per_step = elapsed_time / self.num_timesteps
                remaining_steps = self.total_timesteps - self.num_timesteps
                remaining_time = remaining_steps * time_per_step
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            else:
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Just started")
            self.last_percent = percent

        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Humanoid Stand Training')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'], 
                       help='Mode: train or evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (for evaluation)')
    parser.add_argument('--total_timesteps', type=int, default=500000,
                       help='Total timesteps for training')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--num_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run on (auto, cpu, or cuda)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    
    return parser.parse_args()


def make_env():
    """Create a function that returns a DMCWrapper environment with waving disabled."""
    def _init():
        return DMCWrapper(enable_waving=False)  # Disable waving for standing only
    return _init


def train_humanoid_stand(total_timesteps=500000, output_dir='results', num_envs=8, device='auto', use_wandb=False):
    """Train the humanoid to stand with parallel environments and optional wandb logging."""
    if use_wandb:
        wandb.init(project="humanoid-stand-wave", entity="andrewkent", config={
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "device": device,
        })

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
        name_prefix="humanoid_stand"
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
    final_model_path = os.path.join(output_dir, "humanoid_stand_final.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # For evaluation, we need a non-vectorized environment
    eval_env = DMCWrapper(enable_waving=False)
    
    # Evaluate the model
    evaluate_model(eval_env, model)
    
    # Record videos
    print("Recording videos of the trained humanoid...")
    video_path = os.path.join(output_dir, f"humanoid_stand_{timestamp}.mp4")
    record_video(eval_env, model, video_path)
    
    # Record a close-up of the standing motion
    closeup_path = os.path.join(output_dir, f"humanoid_stand_closeup_{timestamp}.mp4")
    record_closeup_video_headless(eval_env, model, closeup_path)
    
    if use_wandb:
        # Log videos to wandb
        wandb.log({
            "video_full": wandb.Video(video_path, fps=30, format="mp4"),
            "video_closeup": wandb.Video(closeup_path, fps=30, format="mp4"),
        })
    
    print(f"Training and evaluation complete.")
    print(f"Full video: {video_path}")
    print(f"Close-up video: {closeup_path}")
    
    if use_wandb:
        wandb.finish()
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        # Train mode
        train_humanoid_stand(
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            num_envs=args.num_envs,
            device=args.device,
            use_wandb=args.wandb
        )
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        
        # Load model
        model = PPO.load(args.model_path)
        
        # Create environment with waving disabled
        env = DMCWrapper(enable_waving=False)
        
        # Evaluate
        evaluate_model(env, model)
        
        # Record videos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(args.output_dir, f"humanoid_stand_{timestamp}.mp4")
        record_video(env, model, video_path)
        
        # Record a close-up of the standing motion
        closeup_path = os.path.join(args.output_dir, f"humanoid_stand_closeup_{timestamp}.mp4")
        record_closeup_video_headless(env, model, closeup_path)
        
        if args.wandb:
            wandb.init(project="humanoid-stand-wave", entity="andrewkent", config={"mode": "evaluate"})
            wandb.log({
                "video_full": wandb.Video(video_path, fps=30, format="mp4"),
                "video_closeup": wandb.Video(closeup_path, fps=30, format="mp4"),
            })
            wandb.finish()
        
        print(f"Evaluation complete.")
        print(f"Full video: {video_path}")
        print(f"Close-up video: {closeup_path}")


if __name__ == "__main__":
    main()
