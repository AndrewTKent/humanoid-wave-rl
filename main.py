"""
Main script for humanoid wave training and evaluation.
"""

import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dmc_wrapper import DMCWrapper
from src.visualization import evaluate_model, record_video


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
    
    return parser.parse_args()


def train_humanoid_wave(total_timesteps=1000000, output_dir='results'):
    """Train the humanoid to stand and wave."""
    # Create the environment
    env = DMCWrapper()
    
    # Verify the environment
    check_env(env)
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=os.path.join(output_dir, f"checkpoints_{timestamp}"),
        name_prefix="humanoid_wave"
    )
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "humanoid_wave_final.zip")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Evaluate the model
    evaluate_model(env, model)
    
    # Record a video
    video_path = os.path.join(output_dir, f"humanoid_wave_{timestamp}.mp4")
    record_video(env, model, video_path)
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        # Train mode
        train_humanoid_wave(
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir
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
        
        # Record video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(args.output_dir, f"humanoid_wave_{timestamp}.mp4")
        record_video(env, model, video_path)


if __name__ == "__main__":
    main()
