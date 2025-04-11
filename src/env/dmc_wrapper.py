"""
DMC Wrapper for Humanoid Environment.

This module provides a wrapper that converts dm_control environments 
to be compatible with Gymnasium interface.
"""

import gymnasium as gym
import numpy as np
from dm_control import suite


class DMCWrapper(gym.Env):
    """
    Wrapper for dm_control environments to make them compatible with Gymnasium.
    
    This wrapper specifically supports the humanoid standing task and adds
    custom reward shaping for a waving behavior.
    """
    
    def __init__(self, domain_name="humanoid", task_name="stand"):
        """
        Initialize the DMC wrapper.
        
        Args:
            domain_name (str): The domain name in dm_control suite
            task_name (str): The task name in dm_control suite
        """
        # Load the environment
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        
        # Get observation specs and determine total dimensions
        obs_spec = self.env.observation_spec()
        total_obs_dim = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))
        
        # Define the observation space (continuous vector of all observations combined)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # Get action specs
        action_spec = self.env.action_spec()
        
        # Define the action space
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        # Keep track of arm position for waving detection
        self.prev_arm_positions = None
        self.prev_delta = 0.0
        self.wave_counter = 0
        self.direction_changes = 0
        
        # Identify right arm joints (based on humanoid model experimentation)
        # These indices control the right arm movement in the humanoid model
        self.right_arm_joint_indices = [5, 6, 7]  # Shoulder and elbow joints
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            observation (np.ndarray): Initial observation
            info (dict): Additional information
        """
        if seed is not None:
            # Set random seed if provided
            super().reset(seed=seed)
            
        # Reset the dm_control environment
        time_step = self.env.reset()
        
        # Reset wave tracking variables
        self.prev_arm_positions = None
        self.prev_delta = 0.0
        self.wave_counter = 0
        self.direction_changes = 0
        
        # Return flattened observation and empty info dict (gym standard)
        return self._flatten_obs(time_step.observation), {}
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            observation (np.ndarray): New observation
            reward (float): Reward for this step
            done (bool): Whether the episode is done
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information
        """
        # Execute action in the dm_control environment
        time_step = self.env.step(action)
        
        # Get flattened observation
        obs = self._flatten_obs(time_step.observation)
        
        # Get the stand reward (or 0 if None)
        stand_reward = float(time_step.reward) if time_step.reward is not None else 0.0
        
        # Calculate wave reward
        wave_reward = self._compute_wave_reward(time_step.observation)
        
        # Combine rewards with appropriate weights
        # Initially prioritize standing, then gradually add waving reward
        progress_factor = min(1.0, self.wave_counter / 1000)  # Curriculum learning
        total_reward = stand_reward + 0.3 * progress_factor * wave_reward
        
        # Check if episode is done
        done = time_step.last()
        
        # Additional info dict with reward components
        info = {
            'stand_reward': stand_reward,
            'wave_reward': wave_reward
        }
        
        # Gym requires both terminated and truncated flags
        truncated = False
        
        self.wave_counter += 1
        
        return obs, total_reward, done, truncated, info
        
    def _flatten_obs(self, obs_dict):
        """
        Flatten the observation dictionary into a 1D numpy array.
        
        Args:
            obs_dict (dict): Observation dictionary from dm_control
            
        Returns:
            np.ndarray: Flattened observation vector
        """
        # Concatenate all observation values into a single vector
        return np.concatenate([
            np.array(v, dtype=np.float32).flatten() 
            for v in obs_dict.values()
        ])
    
    def _compute_wave_reward(self, observation):
        """
        Compute reward for wave-like motion of the right arm.
        
        Args:
            observation (dict): Observation dictionary from dm_control
            
        Returns:
            float: Wave reward value
        """
        # Extract right arm joint positions
        joint_angles = observation['joint_angles']
        arm_positions = joint_angles[self.right_arm_joint_indices]
        
        # Initialize wave reward
        wave_reward = 0.0
        
        # First time step, just store the position
        if self.prev_arm_positions is None:
            self.prev_arm_positions = arm_positions.copy()
            self.prev_delta = 0.0
            return wave_reward
        
        # Calculate joint movement (focus on shoulder joint)
        shoulder_delta = arm_positions[0] - self.prev_arm_positions[0]
        
        # Detect direction change (essential for waving)
        # We're looking for oscillatory movement
        if self.prev_arm_positions[0] > 0.2:  # If arm is elevated
            if (shoulder_delta > 0.05 and self.prev_delta < -0.05) or \
               (shoulder_delta < -0.05 and self.prev_delta > 0.05):
                # Direction changed significantly
                self.direction_changes += 1
                wave_reward += 1.0  # Reward for direction change while elevated
        
        # Reward for keeping arm elevated (above horizontal)
        if arm_positions[0] > 0.3:
            wave_reward += 0.2
        
        # Extra reward for multiple direction changes (sustained waving)
        wave_reward += 0.1 * min(10, self.direction_changes)
        
        # Store current positions for next step
        self.prev_arm_positions = arm_positions.copy()
        self.prev_delta = shoulder_delta
        
        return wave_reward
