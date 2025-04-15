import gymnasium as gym
import numpy as np
from dm_control import suite
import collections

class DMCWrapper(gym.Env):
    """Wrapper for humanoid standing with focus on head height and feet on ground."""

    def __init__(self, domain_name="humanoid", task_name="stand",
                 max_steps=1000, seed=None, 
                 lying_down_threshold=30,  # Number of steps before terminating when lying down
                 init_randomization=0.05,   # Amount of randomization in initial pose (0-1)
                 action_smoothing=0.2):     # Amount of action smoothing (0-1)

        # Load the environment
        random_state = np.random.RandomState(seed) if seed is not None else None
        self.env = suite.load(domain_name=domain_name, task_name=task_name, 
                             task_kwargs={'random': random_state})

        # Get observation specs
        obs_spec = self.env.observation_spec()
        if not isinstance(obs_spec, collections.OrderedDict):
            obs_spec = collections.OrderedDict(obs_spec)
        
        self._obs_keys = list(obs_spec.keys())
        total_obs_dim = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float64
        )

        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )

        # Episode tracking
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0
        self.lying_down_steps = 0  # Track how long agent has been lying down
        self.lying_down_threshold = lying_down_threshold  # Threshold for early termination
        self.init_randomization = init_randomization  # Amount of randomization in initial pose
        
        # Metadata for rendering
        self.metadata = {'render_modes': ['human', 'rgb_array'], 
                         'render_fps': int(1.0 / self.env.physics.timestep())}
        
        # Track rewards for debugging
        self.total_reward = 0.0
        
        # Track previous height for velocity calculation
        self.prev_height = None
        
        # New: For action smoothing
        self.action_smoothing = action_smoothing
        self.last_action = None
        
        # New: For observation normalization
        self.obs_running_mean = None
        self.obs_running_var = None
        self.obs_epsilon = 1e-8
        
        # New: For recovery reward
        self.was_falling = False

    def _normalize_obs(self, obs):
        """Normalize observations using running statistics."""
        if self.obs_running_mean is None:
            self.obs_running_mean = np.zeros_like(obs)
            self.obs_running_var = np.ones_like(obs)
            return obs
        
        # Update running mean and variance
        batch_mean = np.mean(obs)
        batch_var = np.var(obs)
        batch_count = 1
        
        delta = batch_mean - self.obs_running_mean
        tot_count = 1 + batch_count
        
        self.obs_running_mean += delta * batch_count / tot_count
        m_a = self.obs_running_var * 1
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * 1 * batch_count / tot_count
        self.obs_running_var = M2 / tot_count
        
        # Normalize
        return (obs - self.obs_running_mean) / (np.sqrt(self.obs_running_var) + self.obs_epsilon)

    def reset(self, seed=None, options=None):
        """Reset the environment with a standard starting position."""
        if seed is not None:
            super().reset(seed=seed)
            if hasattr(self.env._task, 'random'):
                self.env._task.random.seed(seed)

        time_step = self.env.reset()

        # Apply starting position
        with self.env.physics.reset_context():
            qpos = self.env.physics.data.qpos.copy()
            qvel = self.env.physics.data.qvel.copy()
            
            # Set to a standing position with optional randomization
            self._apply_standing_position(qpos, qvel, randomize=self.init_randomization)
            
            # Apply the modified state
            self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset episode variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self._get_height()
        self.total_reward = 0.0
        self.lying_down_steps = 0
        self.prev_height = self._get_height()
        self.last_action = None
        self.was_falling = False

        # Get initial observation and info
        obs = self._flatten_obs(time_step.observation)
        
        # Normalize observation
        norm_obs = self._normalize_obs(obs)
        
        # Get foot heights for info
        foot_heights = self._get_foot_heights()
        
        info = {'height': self._get_height(),
                'foot_heights': foot_heights}

        return norm_obs.astype(np.float32), info
    
    def _apply_standing_position(self, qpos, qvel, randomize=0.0):
        """Apply a stable standing position with feet firmly on ground.
        
        Args:
            qpos: Position state to modify
            qvel: Velocity state to modify
            randomize: Amount of randomization to apply (0.0 = none, 1.0 = maximum)
        """
        # Set to standing height
        target_height = 1.35  # Target humanoid height
        qpos[2] = target_height
        
        # Set perfect upright orientation with slight randomization
        if randomize > 0.0:
            # Small random perturbation to orientation (mainly around vertical axis)
            # w component stays close to 1 for stability
            w = 1.0 - randomize * 0.1 * np.random.random()  # Keep w close to 1
            # Small random values for x,y components (pitch/roll)
            x = randomize * 0.1 * (np.random.random() - 0.5)
            y = randomize * 0.1 * (np.random.random() - 0.5)
            # Slightly larger randomization for z component (yaw)
            z = randomize * 0.2 * (np.random.random() - 0.5)
            
            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            qpos[3:7] = [w/norm, x/norm, y/norm, z/norm]
        else:
            qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Perfect upright quaternion
        
        # Set balanced joint positions - slight knee bend for stability
        joint_indices = slice(7, len(qpos))
        qpos[joint_indices] = 0.0  # Zero all joints first
        
        # Try to set specific joint angles if available
        try:
            # These are approximate - adjust for your specific model
            knee_indices = [self.env.physics.model.name2id('left_knee', 'joint'),
                          self.env.physics.model.name2id('right_knee', 'joint')]
            knee_indices = [i-7 for i in knee_indices if i != -1]  # Adjust to local indices
            
            # Find hip indices if available
            hip_indices = []
            for joint_name in ['left_hip_x', 'left_hip_y', 'left_hip_z', 
                              'right_hip_x', 'right_hip_y', 'right_hip_z']:
                try:
                    idx = self.env.physics.model.name2id(joint_name, 'joint')
                    if idx != -1:
                        hip_indices.append(idx-7)  # Adjust to local indices
                except:
                    pass
                    
            # Find ankle indices if available
            ankle_indices = []
            for joint_name in ['left_ankle', 'right_ankle']:
                try:
                    idx = self.env.physics.model.name2id(joint_name, 'joint')
                    if idx != -1:
                        ankle_indices.append(idx-7)  # Adjust to local indices
                except:
                    pass
            
            # Set knee bend
            for idx in knee_indices:
                if idx >= 0:
                    # Base knee bend with optional randomization
                    base_knee_angle = 0.1
                    if randomize > 0.0:
                        knee_random = randomize * 0.05 * np.random.random()
                        qpos[joint_indices][idx] = base_knee_angle + knee_random
                    else:
                        qpos[joint_indices][idx] = base_knee_angle
                        
            # Set slight hip flexion for better balance
            for idx in hip_indices:
                if idx >= 0:
                    # Only apply to y-axis (forward/backward) hip joints
                    if 'hip_y' in self.env.physics.model.id2name(idx+7, 'joint'):
                        base_hip_angle = 0.05  # Slight forward lean
                        if randomize > 0.0:
                            hip_random = randomize * 0.03 * (np.random.random() - 0.5)
                            qpos[joint_indices][idx] = base_hip_angle + hip_random
                        else:
                            qpos[joint_indices][idx] = base_hip_angle
            
            # Set slight ankle flexion for stable base
            for idx in ankle_indices:
                if idx >= 0:
                    base_ankle_angle = -0.05  # Slight backward ankle angle to balance knee bend
                    if randomize > 0.0:
                        ankle_random = randomize * 0.03 * (np.random.random() - 0.5)
                        qpos[joint_indices][idx] = base_ankle_angle + ankle_random
                    else:
                        qpos[joint_indices][idx] = base_ankle_angle
            
            # New: Add more complex joint randomization for diverse poses
            if randomize > 0.1:  # Only apply for higher randomization levels
                # Randomize all joint angles slightly
                all_joints = slice(7, len(qpos))
                qpos[all_joints] += randomize * 0.1 * np.random.randn(len(qpos[all_joints]))
        
        except Exception as e:
            pass  # If joint lookup fails, continue with default pose
        
        # Zero all velocities with optional tiny random values for robustness
        if randomize > 0.0:
            qvel[:] = randomize * 0.01 * np.random.randn(*qvel.shape)
        else:
            qvel[:] = 0.0

    def step(self, action):
        """Take a step with focus on head height and feet position."""
        action = action.astype(np.float64)
        
        # New: Apply action smoothing
        if self.last_action is not None and self.action_smoothing > 0:
            action = self.action_smoothing * self.last_action + (1 - self.action_smoothing) * action
        
        self.last_action = action.copy()
        
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Get observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        # Normalize observation
        norm_obs = self._normalize_obs(obs_flat)
        
        # New: Track previous falling state for recovery detection
        self.was_falling = self.lying_down_steps > 5
        
        # Calculate reward focused on head height and feet position
        reward = self._compute_stand_reward()
        
        # New: Add recovery reward
        recovery_reward = 0
        if self.was_falling and self._get_height() > 0.8:  # If was falling but now upright
            recovery_reward = 5.0  # Big bonus for recovery!
            print(f"Recovery bonus awarded: {recovery_reward}")
        
        # Add recovery reward to total
        reward += recovery_reward
        
        self.total_reward += reward
        
        # Determine if done
        terminated = time_step.last()
        
        # Get current heights
        current_height = self._get_height()
        foot_heights = self._get_foot_heights()
        
        # Early termination for falling
        if current_height < 0.3:
            self.lying_down_steps += 1
        else:
            self.lying_down_steps = 0
            
        # Terminate if lying down for too long
        fall_terminated = self.lying_down_steps >= self.lying_down_threshold
        # Update terminated state if the agent has fallen
        terminated = terminated or fall_terminated

        # Check for jumping (feet too high off ground)
        jump_detected = False
        
        # Regular truncation for max steps
        truncated = self.steps_this_episode >= self.max_steps
        
        # Track best height
        if current_height > self.best_height_this_episode:
            self.best_height_this_episode = current_height
            
        # Store current height for next step
        self.prev_height = current_height
            
        # Prepare info dictionary
        info = {
            'height': current_height,
            'foot_heights': foot_heights,
            'steps': self.steps_this_episode,
            'fall_terminated': fall_terminated,
            'jump_detected': jump_detected,
            'total_reward': self.total_reward,
            'best_height': self.best_height_this_episode,
            'recovery_reward': recovery_reward
        }
        
        # Add final info for episode end
        if terminated or truncated:
            info['final_info'] = {
                'height': current_height,
                'foot_heights': foot_heights,
                'max_height': self.best_height_this_episode,
                'steps': self.steps_this_episode,
                'total_reward': self.total_reward,
                'terminal_observation': norm_obs,
            }
        
        # Ensure reward is a scalar
        reward = float(reward)
        
        return norm_obs, reward, terminated, truncated, info

    def _get_height(self):
        """Get current torso height."""
        if hasattr(self.env.physics, 'torso_height'):
            return self.env.physics.torso_height()
        return self.env.physics.named.data.geom_xpos['torso', 'z']
    
    def _get_foot_heights(self):
        """Get the heights of the feet."""
        try:
            left_foot_height = self.env.physics.named.data.geom_xpos['left_foot', 'z']
            right_foot_height = self.env.physics.named.data.geom_xpos['right_foot', 'z']
            return [float(left_foot_height), float(right_foot_height)]
        except:
            return [0.1, 0.1]  # Default if lookup fails
            
    def _compute_stand_reward(self):
        """Improved reward function that balances height, orientation, feet position, and stability."""
        physics = self.env.physics
        
        # 1. Height Reward Component (more gradual than before)
        try:
            head_height = physics.named.data.geom_xpos['head', 'z']
        except:
            head_height = self._get_height()
        
        # Target standing height
        STAND_HEIGHT = 1.4
        
        # Smoother height reward using tanh-based normalization
        # Maps height from 0 to STAND_HEIGHT to a value between -2 and +2
        # Less harsh penalties, more gradual rewards
        if head_height < 0.05:
            # Very low heights still get significant penalty
            height_reward = -3.0
        else:
            # Normalized height from 0.05 to STAND_HEIGHT
            norm_height = (head_height - 0.05) / (STAND_HEIGHT - 0.05)
            # Clip to avoid extreme values
            norm_height = max(0.0, min(1.0, norm_height))
            # Transform to a reward between -2 and +2 with a smooth gradient
            height_reward = 4.0 * (norm_height - 0.5)
        
        # 2. Orientation Reward Component
        # Get quaternion representing torso orientation (w,x,y,z)
        try:
            # The first quaternion is usually the root/torso
            quat = physics.data.qpos[3:7]
            # Perfect upright orientation is [1,0,0,0]
            # Dot product gives cosine of angle between current and upright
            upright_reward = quat[0]  # w component, gives 1.0 when perfectly upright
            # Scale to a 0-1 range, emphasizing being close to upright
            upright_reward = upright_reward ** 2
        except:
            upright_reward = 0.5  # Default if we can't get orientation
        
        # 3. Feet Position Reward
        foot_heights = self._get_foot_heights()
        foot_reward = 0.0
        
        # Reward for feet being on the ground (not too high)
        foot_ground_contact = sum(1.0 for h in foot_heights if h < 0.1)
        foot_reward += 0.5 * (foot_ground_contact / len(foot_heights))
        
        # 4. Stability Reward (penalize excessive angular velocity)
        try:
            # Get angular velocity norm (how fast the humanoid is rotating)
            ang_vel = physics.data.qvel[3:6]  # Angular velocity components
            ang_vel_norm = np.linalg.norm(ang_vel)
            
            # Penalize high angular velocity (unstable rotation)
            # Using a soft exponential penalty that maxes out at -1.0
            stability_reward = -min(1.0, ang_vel_norm / 5.0)
        except:
            stability_reward = 0.0  # Default if we can't get angular velocity
            
        # New: 5. Energy Efficiency Component
        try:
            # Penalize excessive control force (encourage smoother, more efficient movements)
            energy_penalty = -0.1 * np.sum(np.square(physics.data.ctrl))
        except:
            energy_penalty = 0.0
        
        # New: 6. Height Stability Reward (penalize oscillation)
        height_stability_reward = 0.0
        if self.prev_height is not None:
            height_velocity = abs(self._get_height() - self.prev_height) / physics.timestep()
            height_stability_reward = -0.3 * min(1.0, height_velocity)
            
        # New: 7. ZMP (Zero Moment Point) Stability
        try:
            com_pos = physics.center_of_mass_position()
            com_vel = physics.center_of_mass_velocity()
            gravity = 9.81
            
            # Simple ZMP approximation (distance from center of feet to projected CoM)
            foot_positions = []
            try:
                foot_positions.append(physics.named.data.geom_xpos['left_foot'])
                foot_positions.append(physics.named.data.geom_xpos['right_foot'])
            except:
                # If can't get foot positions, use approximation
                foot_positions = [[0, 0, 0], [0, 0, 0]]
                
            foot_center = np.mean(foot_positions, axis=0)
            
            # Calculate ZMP
            zmp_x = com_pos[0] + com_vel[0] * np.sqrt(com_pos[2] / gravity)
            zmp_y = com_pos[1] + com_vel[1] * np.sqrt(com_pos[2] / gravity)
            
            # Reward for ZMP being close to foot center
            zmp_dist = np.sqrt((zmp_x - foot_center[0])**2 + (zmp_y - foot_center[1])**2)
            zmp_reward = -0.5 * min(1.0, zmp_dist)
        except:
            zmp_reward = 0.0  # Default if calculation fails
        
        # 8. Combine all rewards with appropriate weights
        combined_reward = (
            2.0 * height_reward +         # Weight: 2.0
            1.5 * upright_reward +        # Weight: 1.5
            1.0 * foot_reward +           # Weight: 1.0
            0.5 * stability_reward +      # Weight: 0.5
            energy_penalty +              # New: penalize excessive force
            height_stability_reward +     # New: reward stable height
            zmp_reward                    # New: reward dynamic stability
        )
        
        # Ensure we return a scalar float
        return float(combined_reward)

    def _flatten_obs(self, obs_dict):
        """Flatten observation dictionary."""
        obs_list = []
        for key in self._obs_keys:
            val = obs_dict[key]
            if isinstance(val, (int, float, np.number)):
                obs_list.append(np.array([val], dtype=np.float64))
            else:
                obs_list.append(np.asarray(val, dtype=np.float64).flatten())
        return np.concatenate(obs_list)

    def render(self, mode='human', height=480, width=640, camera_id=None):
        """Render the environment."""
        if camera_id is None:
            camera_id = 0 if mode == 'human' else 2
            
        if mode == 'rgb_array':
            return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        elif mode == 'human':
            return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
