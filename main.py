"""
Main script for humanoid stand & wave training and evaluation with wandb tracking.
Enhanced with curriculum learning, exploration parameters, and learning schedules.
"""

import os
import time
import argparse
import torch
import numpy as np
from datetime import datetime
import wandb
from typing import Callable # Added for type hinting schedule functions

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Assuming DMCWrapper is in src.dmc_wrapper
from src.dmc_wrapper import DMCWrapper
# Assuming evaluate_model and video recording functions are in src.visualization
from src.visualization import evaluate_model #, record_video, record_closeup_video_headless (Using os.system for now)

class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress and logging to wandb.
    Fixed to initialize properly without requiring model to be set.
    """
    def __init__(self, total_timesteps, num_envs=1, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -0.1
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        # Initialize with specified num_envs instead of accessing model.env
        self.current_episode_reward = np.zeros(num_envs)
        self.current_episode_length = np.zeros(num_envs)
        self.max_height_reached = np.zeros(num_envs)
        self.fps_buffer = []
        self.fps_buffer_size = 10  # Average FPS over last 10 updates

    def _on_step(self):
        """Called after each step of the environment"""
        # Accumulate rewards and lengths per environment
        self.current_episode_reward += self.locals['rewards']
        self.current_episode_length += 1

        # Track max heights per environment
        for i in range(len(self.locals['infos'])):
            if 'height' in self.locals['infos'][i]:
                self.max_height_reached[i] = max(self.max_height_reached[i], self.locals['infos'][i]['height'])

            # Check if episode is done for each environment
            if self.locals['dones'][i]:
                info = self.locals['infos'][i].get("final_info", self.locals['infos'][i]) # Handle VecEnv termination signal

                # Log episode metrics to wandb with "episode/" prefix
                log_data = {
                    "episode/reward": self.current_episode_reward[i],
                    "episode/length": self.current_episode_length[i],
                    "episode/stand_reward": info.get('stand_reward', 0),
                    "episode/wave_reward": info.get('wave_reward', 0),
                    "episode/max_height": self.max_height_reached[i],
                    "episode/standing_assist": info.get('standing_assist', 0),
                    "episode/early_termination_reason": info.get('early_termination_reason', "none"),
                    "timesteps": self.num_timesteps,
                }
                if wandb.run: # Check if wandb is active
                    wandb.log(log_data)

                # Reset accumulators for the finished environment
                self.current_episode_reward[i] = 0
                self.current_episode_length[i] = 0
                self.max_height_reached[i] = 0

        # Calculate and log progress (based on total steps)
        percent = round(100 * self.num_timesteps / self.total_timesteps, 1)
        if percent > self.last_percent + 0.9: # Log every ~1%
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
        """Called after each policy update / rollout collection."""
        if not wandb.run:
             return

        # Log all SB3 training metrics automatically captured by the logger
        metrics = self.logger.name_to_value if hasattr(self.logger, 'name_to_value') else {}

        log_data = {}
        for key, value in metrics.items():
            # Standardize keys for wandb
            if key.startswith("rollout/") or key.startswith("train/") or key.startswith("time/"):
                log_data[key] = value
            else: # Default to train/ prefix if unsure
                log_data[f"train/{key}"] = value

        # Add current learning rate (might be a schedule)
        if hasattr(self.model, 'lr_schedule') and hasattr(self.model, '_current_progress_remaining'):
            log_data["train/learning_rate"] = self.model.lr_schedule(self.model._current_progress_remaining)
        
        # Add current clip range (might be a schedule)
        if hasattr(self.model, 'clip_range') and hasattr(self.model, '_current_progress_remaining'):
            log_data["train/clip_range"] = self.model.clip_range(self.model._current_progress_remaining)

        log_data["timesteps"] = self.num_timesteps
        wandb.log(log_data)


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).

        :param progress_remaining: Remaining progress factor
        :return: current learning rate
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Humanoid Stand and Wave Training with Curriculum Learning")
    
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
    
    # Environment settings
    parser.add_argument('--enable_waving', action='store_true',
                       help='Enable waving behavior in training')
    parser.add_argument('--initial_standing_assist', type=float, default=0.8,
                       help='Initial standing assistance level (0.0-1.0) for curriculum learning')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # Parallelization and hardware
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
                       help='Use linear schedule for learning rate and clip range')
    
    # Network architecture
    parser.add_argument('--net_arch', type=str, default='[256, 256]',
                       help='Network architecture as a Python list of layer sizes')
    
    return parser.parse_args()
    
def make_env(rank: int, seed: int = 0, enable_waving=False, initial_standing_assist=0.8, assist_decay_rate=0.9999, max_steps=1000):
    """
    Utility function for multiprocessed env.

    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :param enable_waving: whether to enable waving reward
    :param initial_standing_assist: initial assist value
    :param assist_decay_rate: decay rate for assist
    :param max_steps: max steps per episode
    """
    def _init():
        env = DMCWrapper(enable_waving=enable_waving,
                         initial_standing_assist=initial_standing_assist,
                         assist_decay_rate=assist_decay_rate, # Pass decay rate
                         max_steps=max_steps)
        # Important: use a different seed for each environment
        env.reset(seed=seed + rank)
        return env
    return _init

def train_humanoid_stand(args):
    """Train the humanoid to stand with parallel environments and optional wandb logging."""
    # Extract args for convenience
    total_timesteps = args.total_timesteps
    output_dir = args.output_dir
    num_envs = args.num_envs
    device = args.device
    use_wandb = args.wandb
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base path for saving results
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
            "enable_waving": args.enable_waving,
            "initial_standing_assist": args.initial_standing_assist
        }
        wandb.init(project="humanoid-stand-wave", entity="andrewkent", config=wandb_config)
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            print("CUDA is available")
            device = "cuda"
        else:
            device = "cpu"
    
    # Report device info
    print(f"Using device: {device}")
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        if num_gpus > 0:
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set a base seed for reproducibility (optional)
    base_seed = np.random.randint(0, 1000000)
    print(f"Using base seed: {base_seed}")
    
    # Create vectorized environment with multiple parallel instances
    print(f"Creating {num_envs} parallel environments...")
    
    # Create environment make function to ensure each env has a different seed
    def make_env(rank):
        """Create env with seed derived from base seed plus rank."""
        def _init():
            env = DMCWrapper(
                domain_name="humanoid",
                task_name="stand",
                enable_waving=args.enable_waving,
                initial_standing_assist=args.initial_standing_assist,
                max_steps=args.max_steps,
                seed=base_seed + rank  # Unique seed per env
            )
            return env
        return _init
    
    # Create the vectorized env
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_envs, 1),  # Save every ~50K steps 
        save_path=os.path.join(base_path, "checkpoints"),
        name_prefix="humanoid",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Set up progress callback - pass num_envs to avoid needing model.env
    progress_callback = ProgressCallback(total_timesteps, num_envs=num_envs)
    
    # Create learning rate and clip range schedules (optional)
    if args.use_linear_schedule:
        # Linear schedule: start at initial value, end at 10% of initial value
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)
        clip_schedule = linear_schedule(0.2, 0.05)  # Standard PPO clip range schedule
    else:
        # Constant values
        lr_schedule = args.learning_rate
        clip_schedule = 0.2  # Standard PPO clip range
    
    # Parse network architecture from args
    try:
        net_arch = eval(args.net_arch)
        if not isinstance(net_arch, list):
            print(f"Warning: Invalid network architecture format. Using default [256, 256]")
            net_arch = [256, 256]
    except:
        print(f"Warning: Failed to parse network architecture. Using default [256, 256]")
        net_arch = [256, 256]
    
    # Create the model
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
    
    # Initialize callbacks with model if they have an init_callback method
    if hasattr(progress_callback, 'init_callback'):
        progress_callback.init_callback(model)
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_callback])
    
    # Save the final model
    final_model_path = os.path.join(base_path, "humanoid_final.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # For evaluation, we need a non-vectorized environment
    eval_env = DMCWrapper(
        domain_name="humanoid",
        task_name="stand",
        enable_waving=args.enable_waving,
        initial_standing_assist=0.0  # No assistance during evaluation
    )
    
    # Evaluate the model
    try:
        eval_results = evaluate_model(eval_env, model)
        print(f"Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Create a default results dictionary so we don't get NoneType errors
        eval_results = {
            "mean_total_reward": -1000.0,
            "mean_stand_reward": -1000.0,
            "mean_wave_reward": 0.0
        }
    
    # Record videos using xvfb-run to handle headless environments
    try:
        print("Recording videos of the trained humanoid...")
        video_path = os.path.join(base_path, f"humanoid_video_{total_timesteps//1000}k.mp4")
        
        # Use render_video.py with xvfb-run for headless rendering
        render_cmd = f"xvfb-run -a python render_video.py --model_path {final_model_path} --output_path {video_path}"
        print(f"Executing: {render_cmd}")
        os.system(render_cmd)
        
        if use_wandb and os.path.exists(video_path):
            # Log video to wandb
            wandb.log({
                "video": wandb.Video(video_path, fps=30, format="mp4"),
                "final_eval/mean_reward": eval_results.get("mean_total_reward", 0),
                "final_eval/mean_stand_reward": eval_results.get("mean_stand_reward", 0), 
                "final_eval/mean_wave_reward": eval_results.get("mean_wave_reward", 0),
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
        enable_waving=args.enable_waving, # Use setting from args for consistency
        initial_standing_assist=0.0, # No assistance during evaluation
        max_steps=args.max_steps
    )

    # Evaluate
    print("Evaluating loaded model...")
    eval_results = evaluate_model(eval_env, model, n_eval_episodes=20) # More episodes for eval
    eval_env.close()
    print(f"Evaluation Results: {eval_results}")


    # Record video using xvfb-run
    try:
        print("Recording evaluation video...")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = os.path.join(args.output_dir, f"humanoid_video_eval_{os.path.basename(args.model_path).replace('.zip','')}.mp4")

        # Assuming render_video.py exists
        render_cmd = f"xvfb-run -a python render_video.py --model_path {args.model_path} --output_path {video_path} --max_steps {args.max_steps} --enable_waving {args.enable_waving}"
        print(f"Executing: {render_cmd}")
        return_code = os.system(render_cmd)
        print(f"Video rendering command finished with code: {return_code}")

        if args.wandb:
            wandb.init(project="humanoid-stand-wave", entity="andrewkent", config=vars(args), job_type="evaluation")
            wandb.log({
                "eval/mean_reward": eval_results["mean_total_reward"],
                "eval/std_reward": eval_results["std_total_reward"],
                "eval/mean_stand_reward": eval_results["mean_stand_reward"],
                "eval/mean_wave_reward": eval_results["mean_wave_reward"],
                "eval/mean_episode_length": eval_results["mean_ep_length"],
            })
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                 print(f"Logging video {video_path} to wandb...")
                 wandb.log({"evaluation_video": wandb.Video(video_path, fps=30, format="mp4")})
            elif not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                 print(f"Video file {video_path} not found or is empty. Skipping wandb log.")
            wandb.finish()

        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error recording or logging video during evaluation: {e}")

    print(f"Evaluation complete.")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        # Train mode
        model, model_path = train_humanoid_stand(args)
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
            
        # Load model
        try:
            model = PPO.load(args.model_path)
            print(f"Model loaded from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Create environment with no standing assistance for proper evaluation
        eval_env = DMCWrapper(
            domain_name="humanoid",
            task_name="stand",
            enable_waving=args.enable_waving,
            initial_standing_assist=0.0  # No assistance during evaluation
        )
        
        # Initialize wandb if requested
        if args.wandb:
            wandb.init(project="humanoid-stand-wave", entity="andrewkent", 
                      config={"mode": "evaluate", "model_path": args.model_path})
        
        # Evaluate
        print("Evaluating model...")
        eval_results = evaluate_model(eval_env, model)
        print(f"Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
        
        # Record video
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(args.output_dir, f"humanoid_eval_{timestamp}.mp4")
            
            # Execute render_video.py script with xvfb-run
            render_cmd = f"xvfb-run -a python render_video.py --model_path {args.model_path} --output_path {video_path}"
            print(f"Executing: {render_cmd}")
            os.system(render_cmd)
            
            if args.wandb and os.path.exists(video_path):
                wandb.log({
                    "video": wandb.Video(video_path, fps=30, format="mp4"),
                    "eval/mean_reward": eval_results.get("mean_total_reward", 0),
                    "eval/mean_stand_reward": eval_results.get("mean_stand_reward", 0),
                    "eval/mean_wave_reward": eval_results.get("mean_wave_reward", 0),
                })
            
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"Error recording video: {e}")
        
        # Finish wandb if active
        if args.wandb:
            wandb.finish()
        
        print("Evaluation complete.")


if __name__ == "__main__":
    main()
