"""
Headless video rendering script for trained humanoid models.
Uses offscreen rendering to work without a display.
"""

import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
from dm_control import suite
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper import mjbindings

# Import Wrapper class - ensure this points to your DMCWrapper class
from src.dmc_wrapper import DMCWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Render video of trained humanoid without display')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='humanoid_stand.mp4',
                       help='Path to save the output video')
    parser.add_argument('--num_frames', type=int, default=500,
                       help='Number of frames to record')
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    return parser.parse_args()

def render_offscreen(physics, height=480, width=640, camera_id=0):
    """Render an image using MuJoCo's offscreen renderer."""
    # Create offscreen buffer if needed
    if not hasattr(physics.contexts.mujoco, 'offwidth'):
        physics.contexts.mujoco.offwidth = width
        physics.contexts.mujoco.offheight = height
        physics.contexts.mujoco.offFBO = None
        physics.contexts.mujoco.offBuffer = np.empty((height, width, 3), dtype=np.uint8)
    
    # Update camera parameters if needed
    if camera_id is not None:
        if camera_id >= physics.model.ncam:
            raise ValueError('Camera ID out of range.')
        mjbindings.mjlib.mjv_defaultCamera(physics.contexts.mjvCamera)
        mujoco_camera = physics.model.cam(camera_id)
        physics.contexts.mjvCamera.fixedcamid = camera_id
        physics.contexts.mjvCamera.type_ = enums.mjtCamera.mjCAMERA_FIXED
    
    # Set viewport size
    viewport = mjbindings.types.MJRRECT(0, 0, width, height)
    
    # Render scene
    mjbindings.mjlib.mjv_updateScene(
        physics.model.ptr, physics.data.ptr, 
        physics.contexts.mjvOption, None, 
        physics.contexts.mjvCamera, 
        enums.mjtCatBit.mjCAT_ALL, 
        physics.contexts.mjvScene)
    
    mjbindings.mjlib.mjr_render(viewport, physics.contexts.mjvScene, physics.contexts.mujoco)
    
    # Read RGB pixels into buffer
    mjbindings.mjlib.mjr_readPixels(
        physics.contexts.mujoco.offBuffer, None, viewport, physics.contexts.mujoco)
    
    return np.flipud(physics.contexts.mujoco.offBuffer).copy()

def record_video_headless(model_path, output_path, num_frames=500, width=640, height=480):
    """Record video of the humanoid in a headless environment"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    print(f"Recording video to {output_path}...")
    
    # Regular environment for action prediction
    env = DMCWrapper()
    
    # Environment with rendering capabilities (don't wrap with pixels)
    render_env = suite.load(domain_name="humanoid", task_name="stand")
    
    # Initialize MuJoCo rendering context
    _ = render_env.physics.contexts
    
    # Reset environments
    obs, _ = env.reset()
    time_step = render_env.reset()
    
    frames = []
    
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  Rendering frame {i+1}/{num_frames}")
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environments
        obs, reward, done, _, _ = env.step(action)
        time_step = render_env.step(action)
        
        # Get frame using offscreen rendering
        frame = render_offscreen(render_env.physics, height=height, width=width, camera_id=0)
        frames.append(frame)
        
        # Reset if done
        if done:
            obs, _ = env.reset()
            time_step = render_env.reset()
    
    # Save the video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    record_video_headless(
        args.model_path, 
        args.output_path, 
        args.num_frames,
        args.width,
        args.height
    )

if __name__ == "__main__":
    main()
