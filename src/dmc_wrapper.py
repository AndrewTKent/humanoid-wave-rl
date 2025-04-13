"""
DMC Wrapper for Humanoid Environment.
"""

import gymnasium as gym
import numpy as np
from dm_control import suite


class DMCWrapper(gym.Env):
    """Wrapper for dm_control environments to make them compatible with Gymnasium."""
    
    def __init__(self):
        # Load the environment
        self.env = suite.load(domain_name="humanoid", task_name="stand")
        
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
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        if seed is not None:
            # Set random seed if provided
            super().reset(seed=seed)
            
        # Reset the dm_control environment
        time_step = self.env.reset()
        
        # Return flattened observation and empty info dict (gym standard)
        return self._flatten_obs(time_step.observation), {}
        
    def step(self, action):
        """Take a step in the environment."""
        # Execute action in the dm_control environment
        time_step = self.env.step(action)
        
        # Get flattened observation
        obs = self._flatten_obs(time_step.observation)
        
        # Get the stand reward (or 0 if None)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        
        # Check if episode is done
        done = time_step.last()
        
        # Additional info dict
        info = {}
        
        # Gym requires both terminated and truncated flags
        truncated = False
        
        return obs, reward, done, truncated, info
        
    def _flatten_obs(self, obs_dict):
        """Flatten the observation dictionary into a 1D numpy array."""
        # Concatenate all observation values into a single vector
        return np.concatenate([
            np.array(v, dtype=np.float32).flatten() 
            for v in obs_dict.values()
        ])
