import gymnasium as gym
import numpy as np
from dm_control import suite
import collections

class DMCWrapper(gym.Env):
    """Simplified wrapper for dm_control humanoid environments focused on standing."""

    def __init__(self, domain_name="humanoid", task_name="stand",
                 initial_standing_assist=0.8, assist_decay_rate=0.9999, 
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

        # Curriculum learning parameters
        self.initial_standing_assist = initial_standing_assist
        self.standing_assist_decay = assist_decay_rate
        self.current_standing_assist = initial_standing_assist

        # Episode tracking
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0

        # Metadata for rendering
        self.metadata = {'render_modes': ['human', 'rgb_array'], 
                         'render_fps': int(1.0 / self.env.physics.timestep())}

    def reset(self, seed=None, options=None):
        """Reset the environment with standing assistance."""
        if seed is not None:
            super().reset(seed=seed)
            if hasattr(self.env._task, 'random'):
                self.env._task.random.seed(seed)

        time_step = self.env.reset()

        # Apply standing assistance
        if self.current_standing_assist > 0.01:
            with self.env.physics.reset_context():
                qpos = self.env.physics.data.qpos.copy()
                qvel = self.env.physics.data.qvel.copy()

                # Raise initial height
                target_height = 1.2 + 0.2 * np.random.uniform(-0.5, 0.5)
                qpos[2] = target_height * self.current_standing_assist + qpos[2] * (1 - self.current_standing_assist)

                # Set torso orientation closer to upright
                noise = (np.random.rand(3) - 0.5) * 0.3 * (1 - self.current_standing_assist)
                qpos[4:7] += noise
                qpos[3:7] /= np.linalg.norm(qpos[3:7])

                # Dampen initial velocities
                qvel[:] *= (1 - self.current_standing_assist)
                
                # Apply modified state
                self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset episode variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self._get_height()

        # Get initial observation and info
        obs = self._flatten_obs(time_step.observation)
        info = {'standing_assist': self.current_standing_assist}

        return obs.astype(np.float32), info

    def step(self, action):
        """Take a step and calculate standing reward."""
        action = action.astype(np.float64)
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Get observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        
        # Calculate stand reward
        reward = self._compute_stand_reward()
        
        # Check termination conditions
        terminated = time_step.last()
        truncated = self.steps_this_episode >= self.max_steps
        
        # Update curriculum on episode end
        if terminated or truncated:
            self.current_standing_assist = max(0.0, self.current_standing_assist * self.standing_assist_decay)
        
        # Track best height
        current_height = self._get_height()
        if current_height > self.best_height_this_episode:
            self.best_height_this_episode = current_height
            
        # Prepare info dictionary
        info = {
            'height': current_height,
            'steps': self.steps_this_episode,
            'standing_assist': self.current_standing_assist,
        }
        
        # Add final info for episode end
        if terminated or truncated:
            info['final_info'] = {
                'height': current_height,
                'max_height': self.best_height_this_episode,
                'steps': self.steps_this_episode,
                'standing_assist': self.current_standing_assist,
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
        """Simplified reward focusing on standing height and orientation."""
        physics = self.env.physics
        current_height = self._get_height()
        
        # Height reward - the main component
        target_height = 1.4
        height_threshold = 0.6
        
        if current_height < height_threshold:
            height_reward = current_height * 0.5
        else:
            height_diff = current_height - height_threshold
            height_reward = 5.0 * (1.0 - np.exp(-3.0 * height_diff))
            
            # Bonus for getting close to target height
            if current_height > target_height * 0.9:
                height_reward += 3.0
        
        # Upright orientation reward
        upright_reward = 0.0
        if hasattr(physics, 'torso_upright'):
            upright_value = physics.torso_upright()
        else:
            # Get torso orientation from physics
            torso_z_axis = physics.named.data.xmat['torso'][6:9]
            upright_value = max(0.0, torso_z_axis[2])
        
        upright_reward = 2.0 * upright_value
        
        # Combine rewards
        total_reward = height_reward + upright_reward
        
        # Scale reward based on difficulty
        curriculum_scale = 1.0 + (1.0 - self.current_standing_assist)
        total_reward *= curriculum_scale
        
        return total_reward

    def _flatten_obs(self, obs_dict):
        """Flatten the observation dictionary into a 1D array."""
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
