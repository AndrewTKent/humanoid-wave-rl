"""
Simplified script for humanoid standing training and evaluation with wandb tracking.
Enhanced with curriculum learning and basic performance monitoring.
"""

import os
import time
import argparse
import torch
import numpy as np
from datetime import datetime
import wandb
from typing import Callable
import re

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import our simplified DMCWrapper
from src.dmc_wrapper import DMCWrapper
# Import evaluation function
from src.visualization import evaluate_model

class ProgressCallback(BaseCallback):
    """
    Simplified callback for tracking training progress and logging to wandb.
    """
    def __init__(self, total_timesteps, num_envs=1, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -0.01
        self.start_time = time.time()
        self.num_envs = num_envs
        self.current_episode_reward = np.zeros(num_envs)
        self.current_episode_length = np.zeros(num_envs)
        self.max_height_reached = np.zeros(num_envs)
        self.fps_buffer = []
        self.fps_buffer_size = 10

    def _on_step(self):
        """Called after each step of the environment"""
        # Accumulate rewards safely
        rewards = self.locals['rewards']
        if isinstance(rewards, (int, float)):
            self.current_episode_reward += rewards
        elif np.isscalar(rewards):
            self.current_episode_reward += float(rewards)
        else:
            rewards_array = np.asarray(rewards)
            if rewards_array.ndim == 1 and len(rewards_array) == self.num_envs:
                self.current_episode_reward += rewards_array
            else:
                try:
                    self.current_episode_reward += rewards_array
                except ValueError:
                    print(f"Warning: Reward broadcasting failed")
                    pass

        # Increment episode length
        self.current_episode_length += 1

        # Track heights per environment
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            for i in range(min(len(self.locals['infos']), self.num_envs)):
                info = self.locals['infos'][i]
                if info is not None and 'height' in info:
                    self.max_height_reached[i] = max(self.max_height_reached[i], info['height'])

                # Check if episode is done for this environment
                if 'dones' in self.locals and i < len(self.locals['dones']) and self.locals['dones'][i]:
                    # Get appropriate info dictionary
                    episode_info = info.get("final_info", info) if info is not None else {}

                    # Log episode metrics to wandb
                    log_data = {
                        "episode/reward": float(self.current_episode_reward[i]),
                        "episode/length": int(self.current_episode_length[i]),
                        "episode/max_height": float(self.max_height_reached[i]),
                        "episode/standing_assist": float(episode_info.get('standing_assist', 0)),
                        "timesteps": self.num_timesteps,
                    }
                    if wandb.run:
                        wandb.log(log_data)

                    # Reset accumulators for the finished environment
                    self.current_episode_reward[i] = 0
                    self.current_episode_length[i] = 0
                    self.max_height_reached[i] = 0

        # Calculate and log progress
        percent = round(100 * self.num_timesteps / self.total_timesteps, 1)
        if percent > self.last_percent + 0.9:  # Log every ~1%
            elapsed_time = time.time() - self.start_time
            if self.num_timesteps > 0:
                time_per_step = elapsed_time / self.num_timesteps
                remaining_steps = self.total_timesteps - self.num_timesteps
                remaining_time = remaining_steps * time_per_step
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | FPS: {fps:.1f} | Elapsed: {elapsed_str} | Remaining: {remaining_str}")

                # Add to FPS buffer for moving average
                self.fps_buffer.append(fps)
                if len(self.fps_buffer) > self.fps_buffer_size:
                    self.fps_buffer.pop(0)
                avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0

                # Log time metrics
                if wandb.run:
                    wandb.log({
                        "time/fps": fps,
                        "time/fps_moving_avg": avg_fps,
                        "time/time_elapsed_s": elapsed_time,
                        "time/total_timesteps": self.num_timesteps,
                        "progress_percent": percent,
                        "timesteps": self.num_timesteps,
                    })
            else:
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Just started")
            self.last_percent = percent

        return True

    def _on_rollout_end(self):
        """Log training metrics after each rollout"""
        if not wandb.run:
            return

        metrics = self.logger.name_to_value if hasattr(self.logger, 'name_to_value') else {}

        log_data = {}
        for key, value in metrics.items():
            if key.startswith("rollout/") or key.startswith("train/") or key.startswith("time/"):
                log_data[key] = value
            else:
                log_data[f"train/{key}"] = value

        # Add current learning rate if applicable
        if hasattr(self.model, 'lr_schedule') and hasattr(self.model, '_current_progress_remaining'):
            log_data["train/learning_rate"] = self.model.lr_schedule(self.model._current_progress_remaining)

        log_data["timesteps"] = self.num_timesteps
        wandb.log(log_data)


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Humanoid Stand Training")
    
    # Mode and output settings
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'], 
                       help='Mode: train or evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (for evaluation)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a saved model to resume training from')
    parser.add_argument('--wandb_id', type=str, default=None,
                       help='Wandb run ID to resume tracking')
    
    # Environment settings
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # Parallelization settings
    parser.add_argument('--num_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run on (auto, cpu, or cuda)')
    
    # PPO algorithm parameters
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate for the optimizer')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Number of steps to run per update')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for updates')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--ent_coef', type=float, default=0.01, 
                       help='Entropy coefficient for exploration')
    parser.add_argument('--use_linear_schedule', action='store_true',
                       help='Use linear schedule for learning rate')
    
    # Network architecture
    parser.add_argument('--net_arch', type=str, default='[256, 256]',
                       help='Network architecture as a Python list of layer sizes')
    
    return parser.parse_args()


def make_env(rank: int, seed: int = 0, max_steps=1000):
    """Utility function for creating environments for parallel running."""
    def _init():
        env = DMCWrapper(
            domain_name="humanoid",
            task_name="stand",
            max_steps=max_steps,
            seed=seed + rank
        )
        return env
    return _init


def train_humanoid_stand(args):
    """Train the humanoid to stand with parallel environments."""
    # Extract args for convenience
    total_timesteps = args.total_timesteps
    output_dir = args.output_dir
    num_envs = args.num_envs
    device = args.device
    use_wandb = args.wandb
    resume_from = args.resume_from
    
    # Determine base path - either from resume path or create a new one
    if resume_from:
        # Extract the directory from resume_from path
        base_path = os.path.dirname(os.path.abspath(resume_from))
        print(f"Resuming training in: {base_path}")
    else:
        # Create timestamp for new run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(output_dir, f"ppo_humanoid_{timestamp}")
        os.makedirs(base_path, exist_ok=True)
        print(f"Results will be saved in: {base_path}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb_config = {
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "device": device,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "net_arch": args.net_arch,
            "max_steps": args.max_steps,
            "resumed_from": resume_from if resume_from else "None"
        }
        
        wandb.init(
            project="humanoid-stand", 
            config=wandb_config,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None
        )
    
    # Determine device
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Set a base seed for reproducibility
    base_seed = np.random.randint(0, 1000000)
    print(f"Using base seed: {base_seed}")
    
    # Create vectorized environment with multiple parallel instances
    print(f"Creating {num_envs} parallel environments...")
    
    env = SubprocVecEnv([
        make_env(
            i, 
            seed=base_seed, 
            max_steps=args.max_steps
        ) for i in range(num_envs)
    ])
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_envs, 1),
        save_path=os.path.join(base_path, "checkpoints"),
        name_prefix="humanoid",
        save_replay_buffer=True,
    )
    
    # Set up progress callback
    progress_callback = ProgressCallback(total_timesteps, num_envs=num_envs)
    
    # Create learning rate schedule if needed
    if args.use_linear_schedule:
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)
        clip_schedule = linear_schedule(0.2, 0.05)
    else:
        lr_schedule = args.learning_rate
        clip_schedule = 0.2
    
    # Parse network architecture
    try:
        net_arch = eval(args.net_arch)
        if not isinstance(net_arch, list):
            print(f"Warning: Invalid network architecture format. Using default [256, 256]")
            net_arch = [256, 256]
    except:
        print(f"Warning: Failed to parse network architecture. Using default [256, 256]")
        net_arch = [256, 256]
    
    # Create or load the model
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        # Load model but don't set the env yet to avoid issues
        model = PPO.load(
            resume_from,
            env=None,  # Don't set env yet
            device=device
        )
        
        # Set environment after loading
        model.set_env(env)
        
        # IMPORTANT FIX: Override the clip_range with the correct type
        # Handle PPO's clip_range correctly whether it's a float or a schedule
        if args.use_linear_schedule:
            # Use a class attribute to store the original value and replace with schedule
            model._original_clip_range = model.clip_range
            model.clip_range = clip_schedule
        else:
            # Make sure it's a float value
            if callable(model.clip_range):
                model._original_clip_range = model.clip_range
                model.clip_range = 0.2
        
        # Similarly for learning rate
        if args.use_linear_schedule:
            model.learning_rate = lr_schedule
        else:
            model.learning_rate = args.learning_rate
    else:
        print(f"Creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=clip_schedule,
            device=device,
            policy_kwargs={
                "net_arch": net_arch,
                "activation_fn": torch.nn.ReLU
            }
        )
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_callback])
    
    # Extract timesteps from existing filename if resuming
    if resume_from:
        # Try to extract previous timesteps from filename
        prev_timesteps = 0
        filename = os.path.basename(resume_from)
        match = re.search(r'(\d+)k', filename)
        if match:
            prev_timesteps = int(match.group(1)) * 1000
        
        # Calculate new total
        new_total_timesteps = prev_timesteps + total_timesteps
        timesteps_str = f"{new_total_timesteps//1000}k"
    else:
        timesteps_str = f"{total_timesteps//1000}k"
    
    # Save the final model
    final_model_path = os.path.join(base_path, f"humanoid_final_{timesteps_str}.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Also save as humanoid_final.zip for backward compatibility
    compat_path = os.path.join(base_path, "humanoid_final.zip")
    model.save(compat_path)
    print(f"Model also saved to {compat_path} for compatibility")
    
    # Evaluate on a non-vectorized environment
    eval_env = DMCWrapper(
        domain_name="humanoid",
        task_name="stand",
        max_steps=args.max_steps
    )
    
    # Evaluate the model
    try:
        eval_results = evaluate_model(eval_env, model)
        print(f"Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        eval_results = {
            "mean_reward": -1000.0,
            "mean_height": 0.0
        }
    
    # Record videos if possible
    try:
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(base_path, f"humanoid_video_{timesteps_str}.mp4")
        
        render_cmd = f"MUJOCO_GL=egl python render_video.py --model_path {final_model_path} --output_path {video_path}"
        print(f"Executing: {render_cmd}")
        os.system(render_cmd)
        
        if use_wandb and os.path.exists(video_path):
            wandb.log({
                "video": wandb.Video(video_path, fps=30, format="mp4"),
                "final_eval/mean_reward": eval_results.get("mean_reward", 0),
                "final_eval/mean_height": eval_results.get("mean_height", 0),
            })
            
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error recording video: {e}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    print(f"Training and evaluation complete.")
    
    return model, final_model_path


def evaluate_trained_model(args):
    """Evaluate a pre-trained model."""
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError("Model path must be provided and exist for evaluation mode")

    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path, device=args.device)

    # Create evaluation environment
    eval_env = DMCWrapper(
        domain_name="humanoid",
        task_name="stand",
        max_steps=args.max_steps
    )

    # Evaluate
    print("Evaluating loaded model...")
    eval_results = evaluate_model(eval_env, model, n_eval_episodes=10)
    eval_env.close()
    print(f"Evaluation Results: {eval_results}")

    # Record video using xvfb-run
    try:
        print("Recording evaluation video...")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = os.path.join(args.output_dir, f"humanoid_eval_{os.path.basename(args.model_path).replace('.zip','')}.mp4")

        render_cmd = f"MUJOCO_GL=egl python render_video.py --model_path {args.model_path} --output_path {video_path} --max_steps {args.max_steps}"
        print(f"Executing: {render_cmd}")
        os.system(render_cmd)

        if args.wandb:
            wandb.init(project="humanoid-stand", config=vars(args), job_type="evaluation")
            wandb.log({
                "eval/mean_reward": eval_results["mean_reward"],
                "eval/mean_height": eval_results.get("mean_height", 0),
                "eval/mean_episode_length": eval_results["mean_ep_length"],
            })
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                 wandb.log({"evaluation_video": wandb.Video(video_path, fps=30, format="mp4")})
            wandb.finish()

        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error recording or logging video: {e}")

    print(f"Evaluation complete.")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        train_humanoid_stand(args)
    elif args.mode == 'evaluate':
        evaluate_trained_model(args)


if __name__ == "__main__":
    main()
