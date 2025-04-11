"""
Training module for humanoid RL agents.

This module provides functions for training RL agents
using PPO from stable-baselines3.
"""

import os
import yaml
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for logging training progress.
    
    Logs training metrics and saves them to a file.
    """
    
    def __init__(self, log_dir, verbose=0):
        """
        Initialize the callback.
        
        Args:
            log_dir (str): Directory to save logs
            verbose (int): Verbosity level
        """
        super(TrainingProgressCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training_log.csv")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file with header
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("timestep,reward,stand_reward,wave_reward,episode_length\n")
    
    def _on_step(self):
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        if self.locals.get("done", False):
            # Log episode results
            ep_info = self.locals.get("ep_info", {})
            if ep_info and "r" in ep_info:
                with open(self.log_file, "a") as f:
                    timestep = self.num_timesteps
                    reward = ep_info.get("r", 0)
                    ep_len = ep_info.get("l", 0)
                    stand_reward = ep_info.get("stand_reward", 0)
                    wave_reward = ep_info.get("wave_reward", 0)
                    
                    f.write(f"{timestep},{reward},{stand_reward},{wave_reward},{ep_len}\n")
        
        return True


def load_config(config_path):
    """
    Load training configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def train_humanoid_wave(env, config, log_dir="./logs", model_dir="./models"):
    """
    Train a humanoid to stand and wave using PPO.
    
    Args:
        env: The environment to train on
        config (dict): Training configuration
        log_dir (str): Directory to save logs
        model_dir (str): Directory to save models
        
    Returns:
        model: Trained PPO model
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract training parameters
    total_timesteps = config.get("total_timesteps", 1000000)
    learning_rate = config.get("learning_rate", 3e-4)
    n_steps = config.get("n_steps", 2048)
    batch_size = config.get("batch_size", 64)
    n_epochs = config.get("n_epochs", 10)
    gamma = config.get("gamma", 0.99)
    checkpoint_freq = config.get("checkpoint_freq", 100000)
    
    # Set up callbacks
    callbacks = [
        TrainingProgressCallback(log_dir=log_dir),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix="humanoid_wave"
        )
    ]
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )
    
    # Train the model
    start_time = time.time()
    print(f"Starting training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "humanoid_wave_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model
