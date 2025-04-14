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
    (Code remains largely the same as provided, ensure wandb logging covers all desired metrics)
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = -0.1
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = np.zeros(self.training_env.num_envs) # Track per env
        self.current_episode_length = np.zeros(self.training_env.num_envs)
        self.max_height_reached = np.zeros(self.training_env.num_envs)
        self.fps_buffer = []
        self.fps_buffer_size = 10  # Average FPS over last 10 updates

    def _on_step(self):
        """Called after each step of the environment"""
        # Accumulate rewards and lengths per environment
        self.current_episode_reward += self.locals['rewards']
        self.current_episode_length += 1

        # Track max heights per environment
        for i in range(self.training_env.num_envs):
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
                    "episode/early_termination": info.get('early_termination', False),
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
        metrics = self.logger.get_log_dict() # Use recommended way to get metrics

        log_data = {}
        for key, value in metrics.items():
            # Standardize keys for wandb
            if key.startswith("rollout/"):
                log_data[key] = value
            elif key.startswith("train/"):
                 log_data[key] = value
            elif key.startswith("time/"): # Already logged in _on_step usually
                 pass # Avoid duplicate logging if covered elsewhere
            else: # Default to train/ prefix if unsure
                 log_data[f"train/{key}"] = value

        # Add current learning rate and clip range (might change with schedules)
        log_data["train/learning_rate"] = self.model.lr_schedule(self.model._current_progress_remaining)
        log_data["train/clip_range"] = self.model.clip_range(self.model._current_progress_remaining)
        # Add current entropy coef (might change with schedules)
        # Note: PPO ent_coef can be a schedule but isn't exposed as easily as lr_schedule post-init
        # We log the initial value in config, logging current value might require accessing internal attributes or recomputing
        # log_data["train/entropy_coef"] = self.model.ent_coef # This might be the initial value or schedule func

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
    parser = argparse.ArgumentParser(description='Humanoid Stand Training')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for evaluation or resuming)')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, # Increased default
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
                        help='Initial learning rate for the optimizer')
    parser.add_argument('--lr_schedule', type=str, default='linear', choices=['const', 'linear'],
                        help='Learning rate schedule type (const or linear)')
    parser.add_argument('--batch_size', type=int, default=256, # Adjusted default
                        help='Batch size for updates (per PPO epoch)')
    parser.add_argument('--n_steps', type=int, default=2048,
                        help='Number of steps to run for each environment per update')
    parser.add_argument('--net_arch', type=str, default='[256, 256]',
                        help='Network architecture as a Python list of layer sizes "[neurons_l1, neurons_l2, ...]"')
    parser.add_argument('--enable_waving', action='store_true',
                        help='Enable waving behavior in training (RECOMMENDED TO KEEP FALSE FOR INITIAL STANDING)')
    parser.add_argument('--initial_standing_assist', type=float, default=0.8,
                        help='Initial standing assistance level (0.0-1.0) for curriculum learning')
    parser.add_argument('--assist_decay_rate', type=float, default=0.9999,
                        help='Decay rate per episode for standing assist (e.g., 0.9999)')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Initial entropy coefficient for exploration')
    parser.add_argument('--ent_schedule', type=str, default='linear', choices=['const', 'linear'],
                        help='Entropy coefficient schedule type (const or linear)')
    parser.add_argument('--max_steps', type=int, default=1500, # Increased default
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility') # Added seed

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

    # --- Setup WandB ---
    if args.wandb:
        # Ensure output dir exists for wandb run dir
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(
            project="humanoid-stand-wave",
            entity="andrewkent", # Replace with your entity if different
            config=vars(args), # Log all arguments
            sync_tensorboard=True, # auto-sync SB3 logs
            monitor_gym=True, # auto-monitor gym environments
            save_code=True, # save the main script to wandb
            dir=args.output_dir, # Save wandb files locally in output dir
        )

    # --- Determine Device ---
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Number of GPUs available: {torch.cuda.device_count()}")


    # --- Create Vectorized Environment ---
    print(f"Creating {args.num_envs} parallel environments...")
    # Set seed for reproducibility if provided
    base_seed = args.seed if args.seed is not None else np.random.randint(0, 1e6)
    print(f"Using base seed: {base_seed}")

    env = SubprocVecEnv([make_env(rank=i,
                                  seed=base_seed,
                                  enable_waving=args.enable_waving,
                                  initial_standing_assist=args.initial_standing_assist,
                                  assist_decay_rate=args.assist_decay_rate, # Pass decay rate
                                  max_steps=args.max_steps)
                         for i in range(args.num_envs)])


    # --- Create Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_humanoid_{timestamp}"
    save_path = os.path.join(args.output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Results will be saved in: {save_path}")


    # --- Setup Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.num_envs, args.n_steps), # Save roughly every 50k total steps
        save_path=checkpoint_dir,
        name_prefix="humanoid_stand_ppo",
        save_replay_buffer=True, # Save buffer if needed for future use (e.g. offline RL)
        save_vecnormalize=True, # Save VecNormalize stats if used (not used here)
    )
    progress_callback = ProgressCallback(args.total_timesteps)


    # --- Prepare Schedules ---
    if args.lr_schedule == 'linear':
        lr_schedule = linear_schedule(args.learning_rate, 1e-5) # Decay to small value
    else:
        lr_schedule = args.learning_rate # Constant

    if args.ent_schedule == 'linear':
        ent_schedule = linear_schedule(args.ent_coef, 0.001) # Decay entropy
    else:
        ent_schedule = args.ent_coef # Constant


    # --- Parse Network Architecture ---
    try:
        net_arch = eval(args.net_arch)
        if not isinstance(net_arch, list): raise ValueError
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=torch.nn.ReLU) # Use separate pi/vf notation
    except Exception as e:
        print(f"Error parsing network architecture '{args.net_arch}': {e}")
        print("Using default architecture [256, 256]")
        net_arch = [256, 256]
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=torch.nn.ReLU)


    # --- Create or Load Model ---
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from: {args.model_path}")
        # Note: When loading, hyperparameters from the saved model are used by default.
        # You might need to pass custom_objects or explicitly set parameters if schedules changed etc.
        # For simplicity, we assume we are either starting fresh or evaluating a finished model.
        # If resuming training with new schedules, it's often better to start fresh or carefully manage loading.
        model = PPO.load(args.model_path, env=env, device=device,
                         # Potentially re-apply schedules if needed when resuming:
                         # learning_rate=lr_schedule,
                         # ent_coef=ent_schedule,
                         # clip_range=0.2 # Add clip schedule if needed
                         )
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1, # Prints training progress summary
            learning_rate=lr_schedule, # Use schedule
            n_steps=args.n_steps, # Num steps per env per update
            batch_size=args.batch_size, # Minibatch size for PPO update epoch
            n_epochs=10, # Num PPO epochs per update
            gamma=0.99, # Discount factor
            gae_lambda=0.95, # Factor for GAE estimation
            clip_range=0.2, # Clipping parameter, can also be scheduled
            ent_coef=ent_schedule, # Use schedule
            vf_coef=0.5, # Value function coefficient
            max_grad_norm=0.5, # Max gradient norm for clipping
            device=device,
            policy_kwargs=policy_kwargs,
            seed=base_seed, # Set seed for the algorithm
            tensorboard_log=os.path.join(save_path, "tensorboard_logs") if args.wandb else None # Log for Tensorboard/WandB
        )

    # Log effective parameters if creating new model and using wandb
    if args.wandb and not (args.model_path and os.path.exists(args.model_path)):
         wandb.config.update({
             "policy_type": type(model.policy).__name__,
             "activation_fn": str(model.policy.activation_fn),
             "effective_batch_size": model.batch_size, # Actual batch size used
             "n_epochs": model.n_epochs,
             "gamma": model.gamma,
             "gae_lambda": model.gae_lambda,
             "clip_range": model.clip_range, # Initial clip range
             "initial_ent_coef": args.ent_coef, # Initial entropy coef
             "vf_coef": model.vf_coef,
             "max_grad_norm": model.max_grad_norm,
             "lr_schedule": args.lr_schedule,
             "ent_schedule": args.ent_schedule,
             "seed": base_seed,
         })


    # --- Train the Model ---
    print(f"Training for {args.total_timesteps} timesteps...")
    print(f"Each PPO update collects {args.num_envs * args.n_steps} timesteps.")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, progress_callback], # Add progress callback
            log_interval=1, # Log TBoard/WandB every update
            reset_num_timesteps=False # Set to False if loading a model to continue timestep count
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")


    # --- Save Final Model ---
    final_model_path = os.path.join(save_path, "humanoid_stand_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # --- Evaluation & Video Recording ---
    print("Evaluating final model...")
    # Create a single eval env with no assistance
    eval_env = DMCWrapper(enable_waving=args.enable_waving,
                          initial_standing_assist=0.0, # NO assist during final eval
                          max_steps=args.max_steps) # Use same max steps for consistency

    eval_results = evaluate_model(eval_env, model, n_eval_episodes=10) # evaluate_model needs to be defined
    eval_env.close()

    print(f"Evaluation Results: {eval_results}")

    if args.wandb:
        # Log evaluation results
        wandb.log({
            "eval/mean_reward": eval_results["mean_total_reward"],
            "eval/std_reward": eval_results["std_total_reward"],
            "eval/mean_stand_reward": eval_results["mean_stand_reward"],
            "eval/mean_wave_reward": eval_results["mean_wave_reward"],
            "eval/mean_episode_length": eval_results["mean_ep_length"],
        })

    # --- Record Video ---
    try:
        print("Recording video of the trained humanoid...")
        video_path = os.path.join(save_path, f"humanoid_video_{args.total_timesteps//1000}k.mp4")

        # Assuming render_video.py exists and takes model/output path
        render_cmd = f"xvfb-run -a python render_video.py --model_path {final_model_path} --output_path {video_path} --max_steps {args.max_steps} --enable_waving {args.enable_waving}"
        print(f"Executing: {render_cmd}")
        return_code = os.system(render_cmd)
        print(f"Video rendering command finished with code: {return_code}")

        if args.wandb and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            print(f"Logging video {video_path} to wandb...")
            wandb.log({
                "video": wandb.Video(video_path, fps=30, format="mp4"), # Assuming render_video saves at ~30fps
            })
        elif not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
             print(f"Video file {video_path} not found or is empty. Skipping wandb log.")

    except Exception as e:
        print(f"Error recording or logging video: {e}")

    env.close() # Close the training envs

    if args.wandb:
        wandb.finish()

    print(f"Training and evaluation complete. Results in {save_path}")
    return final_model_path


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
        train_humanoid_stand(args)
    elif args.mode == 'evaluate':
        evaluate_trained_model(args)

if __name__ == "__main__":
    main()
