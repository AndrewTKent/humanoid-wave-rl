"""
Software-only rendering script for trained humanoid models.
Works in truly headless environments with no OpenGL/GLFW dependencies.
"""

import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
import mujoco

# Import Wrapper class
from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Render video of trained humanoid using software renderer')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_video.mp4',
                       help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                       help='Number of frames to record')
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    return parser.parse_args()

def render_software(model_xml, data, height=480, width=640, camera_id=0):
    """Render an image using MuJoCo's software renderer."""
    # Create renderer
    renderer = mujoco.Renderer(model_xml, height, width)
    
    # Update scene and render
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def load_model_xml():
    """Load the humanoid model XML directly."""
    from dm_control import suite
    env = suite.load(domain_name="humanoid", task_name="stand")
    model_xml = env.physics.model.to_xml()
    return model_xml

def record_video_software(model_path, output_path, num_frames=500, width=640, height=480):
    """Record video of the humanoid using software rendering only."""
    print(f"Loading policy model from {model_path}")
    policy_model = PPO.load(model_path, device="cpu")  # Force CPU to avoid CUDA warnings
    
    print(f"Recording video to {output_path}...")
    
    # Load MuJoCo model XML
    print("Loading MuJoCo model...")
    model_xml = load_model_xml()
    
    # Create MuJoCo model and data
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    
    # Regular environment for action prediction
    env = DMCWrapper()
    obs, _ = env.reset()
    
    # Reset MuJoCo state to match gym env
    mujoco.mj_resetData(model, data)
    
    frames = []
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from policy model
        action, _ = policy_model.predict(obs, deterministic=True)
        
        # Step gym environment to get next observation
        obs, reward, done, _, _ = env.step(action)
        
        # Apply same action to MuJoCo simulation
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        # Render frame using software renderer
        frame = render_software(model, data, height=height, width=width, camera_id=0)
        frames.append(frame)
        
        # Reset if done
        if done:
            obs, _ = env.reset()
            mujoco.mj_resetData(model, data)
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    record_video_software(
        args.model_path, 
        args.output_path, 
        args.num_frames,
        args.width,
        args.height
    )

if __name__ == "__main__":
    main()
