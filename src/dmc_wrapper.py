import gymnasium as gym
import numpy as np
from dm_control import suite
import collections

class DMCWrapper(gym.Env):
    """Wrapper for humanoid standing with focus on head height and feet on ground."""

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
        
        # Track rewards for debugging
        self.total_reward = 0.0
        # Track previous height for velocity calculation
        self.prev_height = None

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
            
            # Set to a standing position
            self._apply_standing_position(qpos, qvel)
            
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
        
        # Get foot heights for info
        foot_heights = self._get_foot_heights()
        
        info = {'height': self._get_height(),
                'foot_heights': foot_heights}

        return obs.astype(np.float32), info
    
    def _apply_standing_position(self, qpos, qvel):
        """Apply a stable standing position with feet firmly on ground."""
        # Set to standing height
        target_height = 1.35  # Target humanoid height
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

    def step(self, action):
        """Take a step with focus on head height and feet position."""
        action = action.astype(np.float64)
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Get observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        
        # Calculate reward focused on head height and feet position
        reward = self._compute_stand_reward()
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
        fall_terminated = False
        if self.lying_down_steps > 20:  # Faster termination when fallen
            fall_terminated = True
            terminated = True

        # Check for jumping (feet too high off ground)
        jump_detected = False
        if max(foot_heights) > 0.4:  # Higher threshold
            jump_detected = True
            # Don't terminate immediately, just note it happened
            info['jump_detected'] = True
            reward -= 20.0  # Add penalty but don't terminate
        
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
            'best_height': self.best_height_this_episode
        }
        
        # Add final info for episode end
        if terminated or truncated:
            info['final_info'] = {
                'height': current_height,
                'foot_heights': foot_heights,
                'max_height': self.best_height_this_episode,
                'steps': self.steps_this_episode,
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
    
    def _get_foot_heights(self):
        """Get the heights of the feet."""
        try:
            left_foot_height = self.env.physics.named.data.geom_xpos['left_foot', 'z']
            right_foot_height = self.env.physics.named.data.geom_xpos['right_foot', 'z']
            return [float(left_foot_height), float(right_foot_height)]
        except:
            return [0.1, 0.1]  # Default if lookup fails

    def _compute_stand_reward(self):
        """Reward function that gives exponentially negative reward below 0.3,
        exponentially positive between 0.3 and STAND_HEIGHT, and zero above."""
        physics = self.env.physics
        
        # Get head height 
        try:
            head_height = physics.named.data.geom_xpos['head', 'z']
        except:
            # If head lookup fails, use torso height as approximation
            head_height = self._get_height()
        
        # Define standard humanoid stand height from DeepMind
        STAND_HEIGHT = 1.4
        
        # Calculate reward based on height
        if head_height < 0.3:
            # Exponentially worse penalty as height approaches 0
            # Base penalty of -30 at height=0.3, gets much worse as height decreases
            c = 30.0  # Base penalty value
            d = 5.0   # Controls how quickly penalty increases
            # This formula gives -30 at height=0.3 and gets exponentially worse as height decreases
            reward = -c * np.exp(d * (0.3 - head_height))
            return float(reward)
        elif head_height <= STAND_HEIGHT:
            # Exponential reward that grows from ~0 at height=0.3 to maximum at height=STAND_HEIGHT
            normalized_height = (head_height - 0.3) / (STAND_HEIGHT - 0.3)
            # Using exponential growth formula: a * (e^(b*x) - 1)
            a = 100.0  # Maximum reward value
            b = 2.0    # Controls curve steepness
            reward = a * (np.exp(b * normalized_height) - 1)
            return float(reward)
        else:
            # Zero reward for heights above STAND_HEIGHT
            return 0.0

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
