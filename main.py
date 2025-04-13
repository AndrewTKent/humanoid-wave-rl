import gymnasium as gym
import numpy as np
from dm_control import suite

class DMCWrapper(gym.Env):
    """Wrapper for dm_control environments to make them compatible with Gymnasium."""
    
    def __init__(self, enable_waving=True):
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
        
        # Flag to enable/disable waving behavior
        self.enable_waving = enable_waving
        
        # Initialize waving-related variables only if waving is enabled
        if self.enable_waving:
            self.prev_arm_positions = None
            self.prev_delta = 0.0
            self.wave_counter = 0
            self.direction_changes = 0
            # Identify right arm joints (based on humanoid model experimentation)
            self.right_arm_joint_indices = [5, 6, 7]  # Shoulder and elbow joints
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        if seed is not None:
            super().reset(seed=seed)
            
        # Reset the dm_control environment
        time_step = self.env.reset()
        
        # Reset wave tracking variables only if waving is enabled
        if self.enable_waving:
            self.prev_arm_positions = None
            self.prev_delta = 0.0
            self.wave_counter = 0
            self.direction_changes = 0
        
        # Return flattened observation and empty info dict (gym standard)
        return self._flatten_obs(time_step.observation), {}
        
    def step(self, action):
        """Take a step in the environment."""
        # Execute action in the dm_control environment
        time_step = self.env.step(action)
        
        # Get flattened observation
        obs = self._flatten_obs(time_step.observation)
        
        # Get the stand reward (or 0 if None)
        stand_reward = float(time_step.reward) if time_step.reward is not None else 0.0
        
        # Compute wave reward only if waving is enabled
        if self.enable_waving:
            wave_reward = self._compute_wave_reward(time_step.observation)
            progress_factor = min(1.0, self.wave_counter / 1000)
            total_reward = stand_reward + 0.3 * progress_factor * wave_reward
            self.wave_counter += 1
        else:
            wave_reward = 0.0
            total_reward = stand_reward
        
        # Check if episode is done
        done = time_step.last()
        
        # Additional info dict with reward components
        info = {
            'stand_reward': stand_reward,
            'wave_reward': wave_reward
        }
        
        # Gym requires both terminated and truncated flags
        truncated = False
        
        return obs, total_reward, done, truncated, info
        
    def _flatten_obs(self, obs_dict):
        """Flatten the observation dictionary into a 1D numpy array."""
        return np.concatenate([
            np.array(v, dtype=np.float32).flatten() 
            for v in obs_dict.values()
        ])
    
    def _compute_wave_reward(self, observation):
        """Compute reward for wave-like motion of the right arm."""
        if not self.enable_waving:
            return 0.0
        
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
        if self.prev_arm_positions[0] > 0.2:  # If arm is elevated
            if (shoulder_delta > 0.05 and self.prev_delta < -0.05) or \
               (shoulder_delta < -0.05 and self.prev_delta > 0.05):
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
