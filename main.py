"""
Main script for humanoid stand & wave training and evaluation with wandb tracking.
Enhanced with curriculum learning and exploration parameters.
"""

import os
import time
import argparse
import torch
import numpy as np
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
        self.current_height = 0
        self.max_height_reached = 0
        self.fps_buffer = []
        self.fps_buffer_size = 10  # Average FPS over last 10 updates

    def _on_step(self):
        """Called after each step of the environment"""
        # Accumulate rewards and lengths for the first environment
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Track heights if available in infos
        if 'height' in self.locals['infos'][0]:
            self.current_height = self.locals['infos'][0]['height']
            self.max_height_reached = max(self.max_height_reached, self.current_height)

        # Check if episode is done for the first environment
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Extract all available info for logging
            info = self.locals['infos'][0]
            
            # Log episode metrics to wandb with "episode/" prefix
            log_data = {
                "episode/reward": self.current_episode_reward,
                "episode/length": self.current_episode_length,
                "episode/stand_reward": info.get('stand_reward', 0),
                "episode/wave_reward": info.get('wave_reward', 0),
                "episode/max_height": self.max_height_reached,
                "episode/standing_assist": info.get('standing_assist', 0),
                "episode/early_termination": info.get('early_termination', False),
                "timesteps": self.num_timesteps,
            }
            
            wandb.log(log_data)
            
            # Reset episode accumulators
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_height = 0
            self.max_height_reached = 0

        # Calculate and log progress
        percent = round(100 * self.num_timesteps / self.total_timesteps, 1)
        if percent > self.last_percent + 0.9:
            elapsed_time = time.time() - self.start_time
            if self.num_timesteps > 0:
                time_per_step = elapsed_time / self.num_timesteps
                remaining_steps = self.total_timesteps - self.num_timesteps
                remaining_time = remaining_steps * time_per_step
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                fps = self.num_timesteps / elapsed_time
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Elapsed: {elapsed_str} | Remaining: {remaining_str}")
                
                # Log time metrics
                wandb.log({
                    "time/fps": fps,
                    "time/time_elapsed": elapsed_time,
                    "time/total_timesteps": self.num_timesteps,
                    "progress_percent": percent,
                    "timesteps": self.num_timesteps,
                })
                
                # Add to FPS buffer for moving average
                self.fps_buffer.append(fps)
                if len(self.fps_buffer) > self.fps_buffer_size:
                    self.fps_buffer.pop(0)
                wandb.log({"time/fps_moving_avg": np.mean(self.fps_buffer)})
                
            else:
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Just started")
            self.last_percent = percent

        return True

    def _on_rollout_end(self):
        """Called after each policy update"""
        # Log all training metrics to wandb with proper prefixes
        metrics = self.logger.name_to_value
        
        # Log all SB3 metrics
        for key, value in metrics.items():
            # Skip non-training metrics
            if key.startswith('time/') or key.startswith('rollout/'):
                continue
                
            # Prefix with train/ if not already prefixed
            if not key.startswith('train/'):
                wandb_key = f"train/{key}"
            else:
                wandb_key = key
                
            wandb.log({wandb_key: value, "timesteps": self.num_timesteps})
        
        # Explicitly log the specific metrics we want to ensure are tracked
        wandb.log({
            "train/explained_variance": metrics.get("explained_variance", 0),
            "train/policy_gradient_loss": metrics.get("policy_gradient_loss", 0),
            "train/value_loss": metrics.get("value_loss", 0),
            "train/entropy_loss": metrics.get("entropy_loss", 0),
            "train/clip_fraction": metrics.get("clip_fraction", 0),
            "train/approx_kl": metrics.get("approx_kl", 0),
            "train/loss": metrics.get("loss", 0),
            "train/std": metrics.get("std", 0),
            "train/n_updates": metrics.get("n_updates", 0),
            "train/learning_rate": metrics.get("learning_rate", 0),
            "train/clip_range": metrics.get("clip_range", 0),
            "time/iterations": metrics.get("n_updates", 0) // 10,  # Assuming 10 updates per iteration
            "timesteps": self.num_timesteps,
        })


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
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for updates (default: auto-selected based on device)')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Number of steps to run per update')
    parser.add_argument('--net_arch', type=str, default='[256, 256]',
                       help='Network architecture as a Python list of layer sizes')
    parser.add_argument('--enable_waving', action='store_true',
                       help='Enable waving behavior in training')
    parser.add_argument('--initial_standing_assist', type=float, default=0.8,
                       help='Initial standing assistance level (0.0-1.0) for curriculum learning')
    parser.add_argument('--ent_coef', type=float, default=0.01, 
                       help='Entropy coefficient for exploration')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    return parser.parse_args()


def make_env(enable_waving=False, initial_standing_assist=0.8, max_steps=1000):
    """Create a function that returns a DMCWrapper environment."""
    def _init():
        return DMCWrapper(enable_waving=enable_waving, 
                         initial_standing_assist=initial_standing_assist,
                         max_steps=max_steps)
    return _init


def train_humanoid_stand(total_timesteps=500000, output_dir='results', num_envs=8, 
                         device='auto', use_wandb=False, learning_rate=3e-4, 
                         batch_size=None, n_steps=2048, net_arch=None, 
                         enable_waving=False, initial_standing_assist=0.8,
                         ent_coef=0.01, max_steps=1000):
    """Train the humanoid to stand with parallel environments and optional wandb logging."""
    if net_arch is None:
        net_arch = [256, 256]
    
    if use_wandb:
        wandb.init(project="humanoid-stand-wave", entity="andrewkent", config={
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "device": device,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "net_arch": net_arch,
            "enable_waving": enable_waving,
            "initial_standing_assist": initial_standing_assist,
            "ent_coef": ent_coef,
            "max_steps": max_steps,
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
    env = SubprocVecEnv([make_env(enable_waving, initial_standing_assist, max_steps) 
                         for _ in range(num_envs)])
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_envs, 1),  # Save every ~50K steps 
        save_path=os.path.join(output_dir, f"checkpoints_{timestamp}"),
        name_prefix="humanoid_stand",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Set up progress callback
    progress_callback = ProgressCallback(total_timesteps)
    
    # Determine optimal batch size if not specified
    if batch_size is None:
        batch_size = 64 if device == "cpu" else 256  # Smaller for CPU, larger for GPU
    
    # Create the model with adaptively scaled exploration
    # High entropy at the beginning, reduces over time
    initial_entropy = ent_coef
    
    # Schedule for entropy coefficient - higher at the start, lower later
    def get_ent_coef():
        progress = model.num_timesteps / total_timesteps
        return initial_entropy * (1.0 - progress * 0.9)  # Reduce by 90% over training
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        ent_coef=ent_coef,  # Start with high entropy for exploration
        clip_range=0.2,
        device=device,
        policy_kwargs={
            "net_arch": net_arch,
            "activation_fn": torch.nn.ReLU
        }
    )
    
    # Log model parameters
    if use_wandb:
        wandb.config.update({
            "policy_type": type(model.policy).__name__,
            "activation_fn": str(model.policy.activation_fn),
            "actual_batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
        })
    
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
    eval_env = DMCWrapper(enable_waving=enable_waving, 
                         initial_standing_assist=0.0)  # No assistance during evaluation
    
    # Evaluate the model
    eval_results = evaluate_model(eval_env, model)
    
    if use_wandb:
        # Log evaluation results
        wandb.log({
            "eval/mean_reward": eval_results["mean_total_reward"],
            "eval/mean_stand_reward": eval_results["mean_stand_reward"],
            "eval/mean_wave_reward": eval_results["mean_wave_reward"],
        })
    
    # Record videos using xvfb-run to handle headless environments
    try:
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(output_dir, f"humanoid_video_{total_timesteps//1000}k.mp4")
        
        # Use render_video.py with xvfb-run for headless rendering
        render_cmd = f"xvfb-run -a python render_video.py --model_path {final_model_path} --output_path {video_path}"
        print(f"Executing: {render_cmd}")
        os.system(render_cmd)
        
        if use_wandb and os.path.exists(video_path):
            # Log video to wandb
            wandb.log({
                "video": wandb.Video(video_path, fps=30, format="mp4"),
            })
            
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error recording video: {e}")
    
    if use_wandb:
        wandb.finish()
    
    print(f"Training and evaluation complete.")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse network architecture from string to list
    try:
        net_arch = eval(args.net_arch)
    except:
        print(f"Error parsing network architecture: {args.net_arch}")
        print("Using default [256, 256] architecture")
        net_arch = [256, 256]
    
    if args.mode == 'train':
        # Train mode
        train_humanoid_stand(
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            num_envs=args.num_envs,
            device=args.device,
            use_wandb=args.wandb,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            net_arch=net_arch,
            enable_waving=args.enable_waving,
            initial_standing_assist=args.initial_standing_assist,
            ent_coef=args.ent_coef,
            max_steps=args.max_steps
        )
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        
        # Load model
        model = PPO.load(args.model_path)
        
        # Create environment with no standing assistance for proper evaluation
        env = DMCWrapper(
            enable_waving=args.enable_waving,
            initial_standing_assist=0.0  # No assistance during evaluation
        )
        
        # Evaluate
        eval_results = evaluate_model(env, model)
        
        # Record video using xvfb-run
        try:
            print("Recording video of the trained humanoid...")
            video_path = os.path.join(args.output_dir, f"humanoid_video_eval.mp4")
            
            # Use render_video.py with xvfb-run for headless rendering
            render_cmd = f"xvfb-run -a python render_video.py --model_path {args.model_path} --output_path {video_path}"
            print(f"Executing: {render_cmd}")
            os.system(render_cmd)
            
            if args.wandb and os.path.exists(video_path):
                wandb.init(project="humanoid-stand-wave", entity="andrewkent", config={"mode": "evaluate"})
                wandb.log({
                    "video": wandb.Video(video_path, fps=30, format="mp4"),
                    "eval/mean_reward": eval_results["mean_total_reward"],
                    "eval/mean_stand_reward": eval_results["mean_stand_reward"],
                    "eval/mean_wave_reward": eval_results["mean_wave_reward"],
                })
                wandb.finish()
                
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"Error recording video: {e}")
        
        print(f"Evaluation complete.")


if __name__ == "__main__":
    main()
