"""
Headless rendering script for humanoid model
"""

import os
import argparse
import tempfile
import numpy as np
import imageio
import mujoco
from stable_baselines3 import PPO
from dm_control import suite
from dm_control.suite import humanoid
from dm_control.rl import control

# Import your wrapper
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
                       help='Camera ID (0 for default view)')
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    return parser.parse_args()

def save_humanoid_xml(xml_path):
    """Save the humanoid XML to a file"""
    # Access the humanoid XML directly
    xml_string = humanoid.STAND_XML
    
    # Write to file
    with open(xml_path, 'w') as f:
        f.write(xml_string)
    
    return xml_path

def render_frame(model, data, width, height, camera=0):
    """Render a frame using MuJoCo's renderer"""
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera)
    return renderer.render()

def main():
    # Parse arguments
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    policy_model = PPO.load(args.model_path, device="cpu")  # Force CPU
    
    print(f"Setting up rendering environment...")
    
    # Save humanoid XML to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_file:
        xml_path = temp_file.name
    
    save_humanoid_xml(xml_path)
    
    # Load the model and create a data instance directly from the file
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Create a gym-wrapped environment for obtaining observations and sending actions
    env = DMCWrapper()
    
    # Reset environment
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
        mujoco.mj_step(model, data)  # Sometimes stepping twice helps with stability
        
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
    
    # Clean up the temporary file
    try:
        os.unlink(xml_path)
    except:
        pass

if __name__ == "__main__":
    main()
