"""
Main script for humanoid standing training and evaluation with optimized performance and wandb logging.
"""

import os
import time
import argparse
import torch
import wandb
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from dmc_wrapper import DMCWrapper
from visualization import evaluate_model, record_video


class WandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to Weights & Biases during training.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.step_rewards = []
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        """
        Log metrics to wandb at each step
        """
        # Accumulate rewards
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
            # Check if any episode has completed
            new_rewards = [
                ep_info["r"] for ep_info in self.model.ep_info_buffer 
                if ep_info["r"] not in self.episode_rewards
            ]
            if len(new_rewards) > 0:
                self.episode_rewards.extend(new_rewards)
                # Log the episode rewards
                for r in new_rewards:
                    wandb.log({
                        "episode/reward": r,
                        "episode/length": 0,  # We don't track this in our env wrapper
                        "global_step": self.num_timesteps
                    })

        # Log training metrics from the model logger
        if self.model.logger is not None and hasattr(self.model.logger, "name_to_value"):
            for k, v in self.model.logger.name_to_value.items():
                # Skip keys that are already being logged
                if k not in ["rollout/ep_rew_mean", "time/fps"]:
                    wandb.log({k: v, "global_step": self.num_timesteps})

        return True


class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress as percentage and estimated time.
    Also logs progress metrics to wandb.
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
        if percent > self.last_percent + 0.09:  # Use 0.09 to account for float precision
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
                
                # Log progress metrics to wandb
                wandb.log({
                    "training/progress_percent": percent,
                    "training/elapsed_seconds": elapsed_time,
                    "training/remaining_seconds": remaining_time,
                    "training/total_seconds": elapsed_time + remaining_time,
                    "global_step": self.num_timesteps
                })
            else:
                # Avoid division by zero at first step
                print(f"Progress: {percent:.1f}% ({self.num_timesteps}/{self.total_timesteps} timesteps) | Just started")
            
            self.last_percent = percent
        
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Humanoid Standing Training')
    
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
    parser.add_argument('--wandb_project', type=str, default='humanoid-stand',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name')
    
    return parser.parse_args()


def make_env():
    """Create a function that returns a DMCWrapper environment."""
    def _init():
        return DMCWrapper()
    return _init


class EvaluationCallback(BaseCallback):
    """
    Callback for running periodic evaluation during training and logging to wandb.
    """
    def __init__(self, eval_env, eval_freq=10000, verbose=0):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        
    def _on_step(self):
        """Run evaluation periodically"""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            # Run evaluation
            print(f"\nRunning evaluation at {self.num_timesteps} timesteps...")
            
            # Run a few episodes and collect rewards
            all_rewards = []
            for _ in range(3):  # Run 3 evaluation episodes
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, _ = self.eval_env.step(action)
                    episode_reward += reward
                
                all_rewards.append(episode_reward)
            
            # Compute statistics
            mean_reward = sum(all_rewards) / len(all_rewards)
            
            # Log to wandb
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/min_reward": min(all_rewards),
                "eval/max_reward": max(all_rewards),
                "global_step": self.num_timesteps
            })
            
            print(f"Evaluation mean reward: {mean_reward:.2f}")
            
            # Update last evaluation step
            self.last_eval_step = self.num_timesteps
            
        return True


def train_humanoid_stand(total_timesteps=1000000, output_dir='results', num_envs=16, device='auto', 
                       wandb_project='humanoid-stand', wandb_entity=None, wandb_run_name=None):
    """Train the humanoid to stand with parallel environments."""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize wandb
    run_name = wandb_run_name or f"humanoid-stand-{timestamp}"
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "device": device,
            "algorithm": "PPO",
            "env": "humanoid-stand"
        }
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            print("CUDA is available, but training may be faster on CPU for MlpPolicy.")
            print("You can force CPU usage with --device cpu or continue with GPU.")
            device = "cuda"  # Still use CUDA by default
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    wandb.config.update({"actual_device": device})
    
    if device == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
            wandb.config.update({f"gpu_{i}": gpu_name})
    
    # Create vectorized environment with multiple parallel instances
    print(f"Creating {num_envs} parallel environments...")
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Create a single environment for evaluation
    eval_env = DMCWrapper()
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1), 
        save_path=os.path.join(output_dir, f"checkpoints_{timestamp}"),
        name_prefix="humanoid_stand"
    )
    
    # Set up progress callback
    progress_callback = ProgressCallback(total_timesteps)
    
    # Set up wandb callback
    wandb_callback = WandbCallback()
    
    # Set up evaluation callback
    eval_callback = EvaluationCallback(eval_env, eval_freq=50000)
    
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
    
    # Log hyperparameters to wandb
    hyperparams = {
        "learning_rate": model.learning_rate,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "policy_network": str(model.policy_kwargs["net_arch"])
    }
    wandb.config.update(hyperparams)
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    print(f"Progress will be shown as iterations, where each iteration processes")
    print(f"{num_envs * model.n_steps} timesteps ({num_envs} envs * {model.n_steps} steps)")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, progress_callback, wandb_callback, eval_callback]
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "humanoid_stand_final.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_stats = evaluate_model(eval_env, model)
    
    # Log final metrics to wandb
    wandb.log({
        "final/mean_reward": eval_stats["mean_reward"],
        "global_step": total_timesteps
    })
    
    # Record videos
    print("Recording videos of the trained humanoid...")
    video_path = os.path.join(output_dir, f"humanoid_stand_{timestamp}.mp4")
    record_video(eval_env, model, video_path)
    
    # Log video to wandb
    wandb.log({"final_video": wandb.Video(video_path, fps=30, format="mp4")})
    
    print(f"Training and evaluation complete.")
    print(f"Video: {video_path}")
    
    # Finish wandb logging
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
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name
        )
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        
        # Initialize wandb for evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.wandb_run_name or f"eval-humanoid-stand-{timestamp}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "mode": "evaluate",
                "model_path": args.model_path
            }
        )
        
        # Load model
        model = PPO.load(args.model_path)
        
        # Create environment
        env = DMCWrapper()
        
        # Evaluate
        eval_stats = evaluate_model(env, model)
        
        # Log evaluation metrics to wandb
        wandb.log({
            "eval/mean_reward": eval_stats["mean_reward"],
        })
        
        # Record videos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(args.output_dir, f"humanoid_stand_{timestamp}.mp4")
        record_video(env, model, video_path)
        
        # Log video to wandb
        wandb.log({"eval_video": wandb.Video(video_path, fps=30, format="mp4")})
        
        print(f"Evaluation complete.")
        print(f"Video: {video_path}")
        
        # Finish wandb logging
        wandb.finish()


if __name__ == "__main__":
    main()
