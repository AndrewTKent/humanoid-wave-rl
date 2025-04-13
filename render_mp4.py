"""
Direct MuJoCo rendering for humanoid models in headless environments.
"""

import os
import argparse
import numpy as np
import imageio
import mujoco
from stable_baselines3 import PPO
from dm_control import suite

# Import DMCWrapper - make sure this is pointing to your class
from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Render video of trained humanoid')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_video.mp4',
                       help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                       help='Number of frames to record')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID (0 for standard view)')
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    return parser.parse_args()

def get_humanoid_model_xml():
    """Extract the XML from DMControl's humanoid model"""
    env = suite.load(domain_name="humanoid", task_name="stand")
    xml_string = env.physics.model.to_xml()
    return xml_string

def render_frame(model, data, width, height, camera=0):
    """Render a frame using MuJoCo's built-in renderer"""
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera)
    return renderer.render()

def main():
    # Parse arguments
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    policy_model = PPO.load(args.model_path, device="cpu")  # Force CPU to avoid CUDA warnings
    
    print(f"Setting up rendering environment...")
    # Get the MuJoCo model XML
    xml_string = get_humanoid_model_xml()
    
    # Load the model and create a data instance
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # Create a regular environment for observations
    env = DMCWrapper()
    
    # Reset both the environment and MuJoCo simulation
    obs, _ = env.reset()
    mujoco.mj_resetData(model, data)
    
    print(f"Recording {args.num_frames} frames to {args.output_path}...")
    frames = []
    
    for i in range(args.num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{args.num_frames}")
        
        # Get action from the policy
        action, _ = policy_model.predict(obs, deterministic=True)
        
        # Step the gym environment to get the next observation
        obs, reward, done, _, _ = env.step(action)
        
        # Apply the same action to our MuJoCo simulation
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        # Render the frame
        frame = render_frame(model, data, args.width, args.height, camera=args.camera_id)
        frames.append(frame)
        
        # Reset if done
        if done:
            print("  Episode done, resetting...")
            obs, _ = env.reset()
            mujoco.mj_resetData(model, data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Save the video
    print(f"Saving video to {args.output_path}...")
    imageio.mimsave(args.output_path, frames, fps=30)
    print(f"Video saved successfully!")

if __name__ == "__main__":
    main()
