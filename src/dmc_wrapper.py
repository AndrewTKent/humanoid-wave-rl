import gymnasium as gym
import numpy as np
from dm_control import suite
import collections

class DMCWrapper(gym.Env):
    """Simplified wrapper focused exclusively on standing up."""

    def __init__(self, domain_name="humanoid", task_name="stand",
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

        # Episode tracking
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0
        self.lying_down_steps = 0  # Track how long agent has been lying down
        
        # Metadata for rendering
        self.metadata = {'render_modes': ['human', 'rgb_array'], 
                         'render_fps': int(1.0 / self.env.physics.timestep())}
        
        # Track rewards
        self.total_reward = 0.0
        self.prev_height = None

    def reset(self, seed=None, options=None):
        """Reset the environment with varied initial poses."""
        if seed is not None:
            super().reset(seed=seed)
            if hasattr(self.env._task, 'random'):
                self.env._task.random.seed(seed)

        time_step = self.env.reset()

        # Apply varied starting pose
        with self.env.physics.reset_context():
            qpos = self.env.physics.data.qpos.copy()
            qvel = self.env.physics.data.qvel.copy()
            
            # Randomly choose between different starting positions
            start_position = np.random.choice(['lying', 'crouched', 'partial'])
            
            if start_position == 'lying':
                # Start lying down - hardest position
                qpos[2] = 0.1  # Very low z position
                
                # Random orientation close to lying flat
                angle = np.pi/2 + np.random.uniform(-0.2, 0.2)  # ~90 degrees
                axis = np.array([1.0, 0.0, 0.0])  # Rotate around x-axis to lie flat
                sin_a = np.sin(angle/2)
                qpos[3:7] = [np.cos(angle/2), axis[0]*sin_a, axis[1]*sin_a, axis[2]*sin_a]
                
                # Random joint positions for lying
                joint_indices = slice(7, len(qpos))
                qpos[joint_indices] = np.random.normal(0, 0.1, size=qpos[joint_indices].shape)
                
                # Zero velocities
                qvel[:] = 0.0
                
            elif start_position == 'crouched':
                # Start in a crouched position - medium difficulty
                qpos[2] = 0.5  # Lower position
                
                # Mostly upright with slight tilt
                qpos[3:7] = [0.95, 0.1, 0.1, 0.0]  # Quaternion with slight tilt
                qpos[3:7] /= np.linalg.norm(qpos[3:7])  # Normalize
                
                # Set joint positions for a crouch
                joint_indices = slice(7, len(qpos))
                qpos[joint_indices] = np.random.normal(0, 0.2, size=qpos[joint_indices].shape)
                
                # Try to bend knees if available
                try:
                    knee_indices = [
                        self.env.physics.model.name2id('left_knee', 'joint'),
                        self.env.physics.model.name2id('right_knee', 'joint')
                    ]
                    knee_indices = [i-7 for i in knee_indices if i != -1]
                    
                    for idx in knee_indices:
                        if idx >= 0:
                            qpos[joint_indices][idx] = 0.6  # Stronger bend
                except:
                    pass
                
                # Zero velocities
                qvel[:] = 0.0
                
            else:  # 'partial'
                # Start partially standing - easier position
                qpos[2] = 0.9  # Higher z position
                
                # Almost upright
                qpos[3:7] = [0.98, 0.0, 0.2, 0.0]  # Quaternion with minimal tilt
                qpos[3:7] /= np.linalg.norm(qpos[3:7])  # Normalize
                
                # Joint positions close to standing
                joint_indices = slice(7, len(qpos))
                qpos[joint_indices] = np.random.normal(0, 0.1, size=qpos[joint_indices].shape)
                
                # Zero velocities
                qvel[:] = 0.0
                
            # Apply the modified state
            self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset episode variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self._get_torso_height()
        self.total_reward = 0.0
        self.lying_down_steps = 0
        self.prev_height = self._get_torso_height()

        # Get initial observation and info
        obs = self._flatten_obs(time_step.observation)
        info = {
            'height': self._get_torso_height(),
            'start_position': start_position
        }

        return obs.astype(np.float32), info
    
    def step(self, action):
        """Take a step with strong focus on standing up and penalties for lying down."""
        action = action.astype(np.float64)
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Get observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        
        # Calculate reward with strong focus on standing up
        reward = self._compute_stand_reward()
        self.total_reward += reward
        
        # Determine if done
        terminated = time_step.last()
        
        # Track height and lying down state
        current_height = self._get_torso_height()
        if current_height < 0.3:  # Clearly lying down
            self.lying_down_steps += 1
        else:
            self.lying_down_steps = 0
            
        # Early termination for prolonged lying down
        fall_terminated = False
        if self.lying_down_steps > 30:  # Terminate faster when lying down
            fall_terminated = True
            terminated = True
        
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
            'steps': self.steps_this_episode,
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
                'total_reward': self.total_reward,
                'terminal_observation': obs_flat,
            }
        
        # Ensure reward is a scalar
        reward = float(reward)
        
        return obs_flat, reward, terminated, truncated, info

    def _get_torso_height(self):
        """Get current torso height."""
        try:
            return self.env.physics.named.data.geom_xpos['torso', 'z']
        except:
            # Fallback if named access fails
            if hasattr(self.env.physics, 'torso_height'):
                return self.env.physics.torso_height()
            return self.env.physics.data.qpos[2]  # Default to root body z position

    def _compute_stand_reward(self):
        """Simplified reward focused on standing up with strong penalties for lying down."""
        physics = self.env.physics
        current_height = self._get_torso_height()
        
        # STRONG penalty for lying down - this creates clear signal about what not to do
        if current_height < 0.3:
            return -100.0  # Severe punishment for lying down
        
        # Significant penalty for being low but not completely flat
        if current_height < 0.7:
            low_height_penalty = -50.0 + (current_height * 50.0)  # Linear scaling from -50 to -15
            return low_height_penalty
        
        # Basic height reward - clear gradient toward standing
        # Exponential scaling to create stronger gradient as height increases
        height_reward = 50.0 * (current_height / 1.4)**2
        
        # Uprightness reward - being vertical is good
        torso_z_axis = physics.named.data.xmat['torso'][6:9]
        upright_value = max(0.0, torso_z_axis[2])  # How aligned with z-axis (0-1)
        upright_reward = 30.0 * (upright_value**2)  # Squared for stronger gradient
        
        # Effort penalty - small penalty for excessive action
        try:
            effort = np.sum(np.square(physics.data.ctrl))
            effort_penalty = -0.5 * effort
        except:
            effort_penalty = 0.0
        
        # Movement reward - reward for getting higher
        height_change = 0.0
        if self.prev_height is not None:
            height_change = current_height - self.prev_height
            if height_change > 0:  # Only reward upward movement
                height_change_reward = 20.0 * height_change
            else:
                height_change_reward = 0.0  # No penalty for downward movement
        else:
            height_change_reward = 0.0
        
        # Bonus for proper standing
        standing_bonus = 0.0
        if current_height > 1.2 and upright_value > 0.9:
            standing_bonus = 100.0  # Big bonus for achieving proper standing
        
        # Combine all reward components
        total_reward = (
            height_reward + 
            upright_reward + 
            effort_penalty + 
            height_change_reward +
            standing_bonus
        )
        
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
