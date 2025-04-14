import gymnasium as gym
import numpy as np
from dm_control import suite
import collections

class DMCWrapper(gym.Env):
    """Simplified wrapper focused exclusively on standing."""

    def __init__(self, domain_name="humanoid", task_name="stand",
                 initial_standing_assist=0.9, assist_decay_rate=0.9995, 
                 max_steps=1000, seed=None):

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

        # Curriculum parameters - slower decay rate for more assistance time
        self.initial_standing_assist = initial_standing_assist  # Higher initial assistance
        self.standing_assist_decay = assist_decay_rate  # Slower decay
        self.current_standing_assist = initial_standing_assist

        # Episode tracking
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0
        self.lying_down_steps = 0  # Track how long agent has been lying down
        
        # Metadata for rendering
        self.metadata = {'render_modes': ['human', 'rgb_array'], 
                         'render_fps': int(1.0 / self.env.physics.timestep())}
        
        # Track rewards for debugging
        self.total_reward = 0.0
        # Track previous height for velocity calculation
        self.prev_height = None

    def reset(self, seed=None, options=None):
        """Reset the environment with different initial poses based on curriculum stage."""
        if seed is not None:
            super().reset(seed=seed)
            if hasattr(self.env._task, 'random'):
                self.env._task.random.seed(seed)

        time_step = self.env.reset()

        # Choose starting position based on curriculum stage
        with self.env.physics.reset_context():
            qpos = self.env.physics.data.qpos.copy()
            qvel = self.env.physics.data.qvel.copy()
            
            # Different starting positions based on curriculum progress
            if self.current_standing_assist > 0.7:
                # Early curriculum: start in a standing position with assistance
                self._apply_standing_position(qpos, qvel)
            elif self.current_standing_assist > 0.4:
                # Mid curriculum: start in a crouched position
                self._apply_crouched_position(qpos, qvel)
            else:
                # Late curriculum: start in random non-lying positions
                self._apply_random_position(qpos, qvel)
                
            # Apply the modified state
            self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset episode variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self._get_height()
        self.total_reward = 0.0
        self.lying_down_steps = 0
        self.prev_height = self._get_height()

        # Get initial observation and info
        obs = self._flatten_obs(time_step.observation)
        info = {'standing_assist': self.current_standing_assist, 
                'height': self._get_height()}

        return obs.astype(np.float32), info
    
    def _apply_standing_position(self, qpos, qvel):
        """Apply a stable standing position with assistance."""
        # Set to standing height
        target_height = 1.35  # Slightly lower than full height for stability
        qpos[2] = target_height
        
        # Set perfect upright orientation
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
            
            for idx in knee_indices:
                if idx >= 0:
                    qpos[joint_indices][idx] = 0.1  # Slight knee bend
        except:
            pass  # If joint lookup fails, continue with default pose
        
        # Zero all velocities
        qvel[:] = 0.0
    
    def _apply_crouched_position(self, qpos, qvel):
        """Apply a crouched starting position."""
        # Lower height
        qpos[2] = 0.8  # Lower position but not on ground
        
        # Slightly tilted but mostly upright
        qpos[3:7] = [0.95, 0.1, 0.0, 0.0]  # Quaternion with slight tilt
        qpos[3:7] /= np.linalg.norm(qpos[3:7])  # Normalize
        
        # Set joint positions for a crouch
        joint_indices = slice(7, len(qpos))
        qpos[joint_indices] = np.random.normal(0, 0.2, size=qpos[joint_indices].shape)
        
        # Try to bend knees and hips if available
        try:
            knee_hip_indices = [
                self.env.physics.model.name2id('left_knee', 'joint'),
                self.env.physics.model.name2id('right_knee', 'joint'),
                self.env.physics.model.name2id('left_hip_x', 'joint'),
                self.env.physics.model.name2id('right_hip_x', 'joint')
            ]
            knee_hip_indices = [i-7 for i in knee_hip_indices if i != -1]
            
            for idx in knee_hip_indices:
                if idx >= 0:
                    qpos[joint_indices][idx] = 0.3  # Stronger bend
        except:
            pass
        
        # Minimal velocities
        qvel[:] = np.random.normal(0, 0.01, size=qvel.shape)
    
    def _apply_random_position(self, qpos, qvel):
        """Apply a random non-lying position."""
        # Random height that's not lying down
        qpos[2] = np.random.uniform(0.5, 1.0)
        
        # Random orientation that's reasonably upright
        angle = np.random.uniform(-0.3, 0.3)
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        sin_a = np.sin(angle/2)
        qpos[3:7] = [np.cos(angle/2), axis[0]*sin_a, axis[1]*sin_a, axis[2]*sin_a]
        
        # Random joint positions
        joint_indices = slice(7, len(qpos))
        qpos[joint_indices] = np.random.normal(0, 0.3, size=qpos[joint_indices].shape)
        
        # Small random velocities
        qvel[:] = np.random.normal(0, 0.05, size=qvel.shape)

    def step(self, action):
        """Take a step with stronger height-based rewards and early termination."""
        action = action.astype(np.float64)
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Get observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        
        # Calculate reward with stronger focus on height
        reward = self._compute_stand_reward()
        self.total_reward += reward
        
        # Determine if done
        terminated = time_step.last()
        
        # Early termination for falling
        current_height = self._get_height()
        # Track lying down state
        if current_height < 0.2:
            self.lying_down_steps += 1
        else:
            self.lying_down_steps = 0
            
        # Terminate if lying down for too long (faster termination)
        fall_terminated = False
        if self.lying_down_steps > 50:  # Terminate faster when lying down
            fall_terminated = True
            terminated = True
        
        # Regular truncation for max steps
        truncated = self.steps_this_episode >= self.max_steps
        
        # Update curriculum on episode end
        if terminated or truncated:
            # Adjust decay based on performance
            if self.best_height_this_episode > 1.2:
                # Very good performance - decay faster
                self.current_standing_assist *= self.standing_assist_decay * 0.9
            elif self.best_height_this_episode > 0.8:
                # Good performance - normal decay
                self.current_standing_assist *= self.standing_assist_decay
            else:
                # Poor performance - decay slower
                self.current_standing_assist *= self.standing_assist_decay * 1.1
            
            # Ensure within bounds
            self.current_standing_assist = max(0.0, min(self.current_standing_assist, 1.0))
            
        # Track best height
        if current_height > self.best_height_this_episode:
            self.best_height_this_episode = current_height
            
        # Store current height for next step
        self.prev_height = current_height
            
        # Prepare info dictionary
        info = {
            'height': current_height,
            'steps': self.steps_this_episode,
            'standing_assist': self.current_standing_assist,
            'fall_terminated': fall_terminated,
            'lying_down_steps': self.lying_down_steps,
            'total_reward': self.total_reward,
            'best_height': self.best_height_this_episode
        }
        
        # Add final info for episode end
        if terminated or truncated:
            info['final_info'] = {
                'height': current_height,
                'max_height': self.best_height_this_episode,
                'steps': self.steps_this_episode,
                'standing_assist': self.current_standing_assist,
                'total_reward': self.total_reward,
                'terminal_observation': obs_flat,
            }
        
        # Ensure reward is a scalar
        reward = float(reward)
        
        return obs_flat, reward, terminated, truncated, info

    def _get_height(self):
        """Get current torso height."""
        if hasattr(self.env.physics, 'torso_height'):
            return self.env.physics.torso_height()
        return self.env.physics.named.data.geom_xpos['torso', 'z']

    def _compute_stand_reward(self):
        """Completely redesigned reward focused on balance and learning to stand."""
        physics = self.env.physics
        current_height = self._get_height()
        
        # Check if feet are on ground (approximate)
        try:
            left_foot_height = physics.named.data.geom_xpos['left_foot', 'z'] 
            right_foot_height = physics.named.data.geom_xpos['right_foot', 'z']
            feet_on_ground = (left_foot_height < 0.1) and (right_foot_height < 0.1)
        except:
            # If foot lookup fails, assume feet are on ground
            feet_on_ground = True
        
        # Massive penalty for lying flat
        if current_height < 0.2:
            return -50.0  # Extremely negative - being flat is catastrophic
        
        # Strong penalty for being very low but not flat
        if current_height < 0.5:
            return -20.0 + current_height * 10.0  # Less penalty as height increases
        
        # Calculate centerline shift (balance indicator)
        try:
            head_pos = physics.named.data.geom_xpos['head', 'x':'y']
            feet_center = (physics.named.data.geom_xpos['left_foot', 'x':'y'] + 
                          physics.named.data.geom_xpos['right_foot', 'x':'y']) / 2.0
            balance_offset = np.linalg.norm(head_pos - feet_center)
            # Balance reward - being well-centered is good
            balance_factor = max(0, 1.0 - balance_offset)
        except:
            # If body part lookup fails, use simplified balance
            balance_factor = 0.5  # Default middle value
        
        # Height reward - exponential with height
        height_factor = 30.0 * (current_height / 1.4)**2 if current_height < 1.4 else 30.0
        
        # Uprightness reward
        torso_z_axis = physics.named.data.xmat['torso'][6:9]
        upright_value = max(0.0, torso_z_axis[2])**2  # Squared for stronger gradient
        upright_reward = 20.0 * upright_value
        
        # Velocity control - reward low velocity when standing, penalize excessive
        vel = physics.data.qvel.copy()
        vel_norm = np.linalg.norm(vel)
        vel_reward = 5.0 * np.exp(-0.5 * vel_norm) if current_height > 1.0 else 0.0
        
        # Upward movement reward - encourage getting up
        height_change = 0
        if self.prev_height is not None:
            height_change = current_height - self.prev_height
            # Only reward significant upward movement
            if height_change > 0.01 and current_height < 1.3:
                height_change_reward = 10.0 * height_change
            else:
                height_change_reward = 0.0
        else:
            height_change_reward = 0.0
        
        # Total reward
        total_reward = (
            height_factor + 
            upright_reward + 
            10.0 * balance_factor + 
            vel_reward + 
            height_change_reward
        )
        
        # If agent is in a good standing position, give big bonus
        if current_height > 1.2 and upright_value > 0.9 and feet_on_ground:
            total_reward += 50.0
        
        return total_reward

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
