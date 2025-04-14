import gymnasium as gym
import numpy as np
from dm_control import suite

class DMCWrapper(gym.Env):
    """Wrapper for dm_control environments to make them compatible with Gymnasium.
    Enhanced with curriculum learning and reward shaping for better standing and waving."""
    
    def __init__(self, enable_waving=True, initial_standing_assist=0.8, max_steps=1000):
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
        
        # Add initial standing assistance (curriculum learning)
        self.initial_standing_assist = initial_standing_assist
        self.standing_assist_decay = 0.9999  # Decay factor per episode
        self.current_standing_assist = initial_standing_assist
        
        # Track progress for early termination
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0
        self.no_progress_steps = 0
        
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
        
        # Apply standing assistance (raise the humanoid partly upright)
        if self.current_standing_assist > 0:
            with self.env.physics.reset_context():
                # Modify initial state to help the humanoid start more upright
                # Slightly randomize starting position for better generalization
                self.env.physics.data.qpos[2] = 1.2 * self.current_standing_assist  # Raise z position
                
                # Set torso orientation to be more upright
                # Quaternion representing upright orientation with small random perturbation
                quat_noise = (np.random.rand(4) - 0.5) * 0.2 * (1 - self.current_standing_assist)
                upright_quat = np.array([1.0, 0.0, 0.0, 0.0]) + quat_noise
                upright_quat = upright_quat / np.linalg.norm(upright_quat)  # Normalize quaternion
                self.env.physics.data.qpos[3:7] = upright_quat
                
                # Also set joint angles to be closer to standing position
                # This is approximate and depends on the specific humanoid model
                for i in range(7, len(self.env.physics.data.qpos)):
                    # Add small random perturbations to joints
                    self.env.physics.data.qpos[i] = np.random.normal(0, 0.1 * (1 - self.current_standing_assist))
                
                # Set zero velocity initially for stability
                self.env.physics.data.qvel[:] = 0
            
            # Decay standing assistance for curriculum learning
            self.current_standing_assist *= self.standing_assist_decay
        
        # Reset progress tracking
        self.steps_this_episode = 0
        self.best_height_this_episode = 0
        self.no_progress_steps = 0
        
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
        # Increment step counter
        self.steps_this_episode += 1
        
        # Execute action in the dm_control environment
        time_step = self.env.step(action)
        
        # Get flattened observation
        obs = self._flatten_obs(time_step.observation)
        
        # Get current height for tracking progress
        current_height = time_step.observation['body_height'][0]
        
        # Track best height achieved this episode
        if current_height > self.best_height_this_episode:
            self.best_height_this_episode = current_height
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        
        # Check for early termination due to lack of progress
        # After 200 steps with no height improvement, terminate if height is still low
        early_termination = False
        if self.no_progress_steps > 200 and current_height < 0.8:
            early_termination = True
        
        # Compute enhanced stand reward with reward shaping
        stand_reward = self._compute_stand_reward(time_step.observation)
        
        # Compute wave reward only if waving is enabled and standing is achieved
        if self.enable_waving and current_height > 1.0:  # Only enable waving when sufficiently upright
            wave_reward = self._compute_wave_reward(time_step.observation)
            # Progressive curriculum: introduce wave reward more strongly as standing improves
            progress_factor = min(1.0, max(0.0, (current_height - 1.0) / 0.5))  # Scale from 0 to 1 as height increases
            wave_weight = 0.3 * min(1.0, self.wave_counter / 1000) * progress_factor
            total_reward = stand_reward + wave_weight * wave_reward
            self.wave_counter += 1
        else:
            wave_reward = 0.0
            total_reward = stand_reward
        
        # Check if episode is done
        done = time_step.last() or early_termination or self.steps_this_episode >= self.max_steps
        
        # Additional info dict with reward components and diagnostics
        info = {
            'stand_reward': stand_reward,
            'wave_reward': wave_reward,
            'height': current_height,
            'steps': self.steps_this_episode,
            'standing_assist': self.current_standing_assist,
            'early_termination': early_termination
        }
        
        # Gym requires both terminated and truncated flags
        truncated = self.steps_this_episode >= self.max_steps
        
        return obs, total_reward, done, truncated, info
        
    def _flatten_obs(self, obs_dict):
        """Flatten the observation dictionary into a 1D numpy array."""
        return np.concatenate([
            np.array(v, dtype=np.float32).flatten() 
            for v in obs_dict.values()
        ])
    
    def _compute_stand_reward(self, observation):
        """Enhanced reward shaping for standing."""
        # Get base reward from DM Control
        stand_reward = float(self.env._task.get_reward(self.env.physics))
        
        # Extract useful state information
        height = observation['body_height'][0]
        velocity = np.linalg.norm(observation['velocity'][0:3])  # Linear velocity
        
        # Add intermediate rewards for partial progress
        height_reward = min(height * 2.0, 2.0)  # Reward for just being higher off the ground
        
        # Orientation reward - reward being upright
        upright_reward = 0.0
        if 'orientation' in observation:
            z_orient = observation['orientation'][2]  # Z component often indicates upright orientation
            upright_reward = max(0.0, z_orient) * 1.5  # Reward for being more upright
        
        # Stability reward - discourage excessive velocity/jittering
        stability_reward = -0.1 * min(velocity, 2.0)  # Penalty for moving too fast
        
        # Joint position/angle reward - encourage natural standing pose
        joint_reward = 0.0
        if 'joint_angles' in observation:
            # Simplified: reward keeping joint angles close to 0 (natural pose)
            joint_angles = observation['joint_angles']
            joint_penalty = np.sum(np.square(joint_angles)) * 0.05
            joint_reward = -min(joint_penalty, 1.0)
        
        # Combined shaped reward
        shaped_reward = stand_reward + height_reward + upright_reward + stability_reward + joint_reward
        
        # Scale reward based on curriculum progress
        curriculum_scale = 1.0 + max(0.0, 2.0 * (1.0 - self.current_standing_assist))
        
        return shaped_reward * curriculum_scale
    
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
        if arm_positions[0] > 0.2:  # If arm is elevated
            # More sensitive detection of direction changes
            if (shoulder_delta > 0.03 and self.prev_delta < -0.03) or \
               (shoulder_delta < -0.03 and self.prev_delta > 0.03):
                self.direction_changes += 1
                wave_reward += 1.0  # Reward for direction change while elevated
        
        # Progressive reward for arm elevation
        # The higher the arm, the better (up to a point)
        elevation_reward = min(max(0, arm_positions[0]) * 3.0, 1.5)
        wave_reward += elevation_reward
        
        # Reward for holding arm out (away from body)
        if abs(arm_positions[1]) > 0.2:  # Second joint controls side movement
            wave_reward += 0.3
        
        # Extra reward for multiple direction changes (sustained waving)
        wave_reward += 0.2 * min(10, self.direction_changes)
        
        # Bonus for frequency - faster waving (if not too fast)
        if 0.05 < abs(shoulder_delta) < 0.2:
            wave_reward += 0.2
        
        # Store current positions for next step
        self.prev_arm_positions = arm_positions.copy()
        self.prev_delta = shoulder_delta
        
        return wave_reward
