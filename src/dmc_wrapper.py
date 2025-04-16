import gymnasium as gym
import numpy as np
from dm_control import suite
import collections


class DMCWrapper(gym.Env):
    """Wrapper for humanoid standing with right arm waving capabilities."""

    def __init__(self, domain_name="humanoid", task_name="stand",
                 max_steps=1000, seed=None, 
                 lying_down_threshold=30,
                 init_randomization=0.05,
                 action_smoothing=0.2,
                 wave_amplitude=0.5,
                 wave_frequency=1.0,
                 wave_reward_weight=0.8,
                 time_reward_scale=0.002):

        random_state = np.random.RandomState(seed) if seed is not None else None
        self.env = suite.load(domain_name=domain_name, task_name=task_name, 
                             task_kwargs={'random': random_state})

        # Configure observation space
        obs_spec = self.env.observation_spec()
        if not isinstance(obs_spec, collections.OrderedDict):
            obs_spec = collections.OrderedDict(obs_spec)
        
        self._obs_keys = list(obs_spec.keys())
        total_obs_dim = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float64
        )

        # Configure action space
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )

        # Core parameters
        self.max_steps = max_steps
        self.lying_down_threshold = lying_down_threshold
        self.init_randomization = init_randomization
        self.action_smoothing = action_smoothing
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_reward_weight = wave_reward_weight
        self.time_reward_scale = time_reward_scale
        
        # Rendering configuration
        self.metadata = {'render_modes': ['human', 'rgb_array'], 
                         'render_fps': int(1.0 / self.env.physics.timestep())}
        
        # State tracking variables
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0
        self.lying_down_steps = 0
        self.total_reward = 0.0
        self.prev_height = None
        self.last_action = None
        self.was_falling = False
        self.wave_time = 0.0
        self.standing_time = 0
        
        # Observation normalization
        self.obs_running_mean = None
        self.obs_running_var = None
        self.obs_epsilon = 1e-8
        
        # Arm joints identification
        self.right_arm_joints = self._identify_right_arm_joints()
        self.right_arm_joint_ids = self._get_right_arm_joint_ids()

    def _identify_right_arm_joints(self):
        """Identify the right arm joints for waving motion."""
        right_arm_patterns = [
            'right_shoulder', 'right_elbow', 'r_shoulder', 'r_elbow',
            'right_arm', 'right_forearm', 'rshoulder', 'relbow'
        ]
        
        identified_joints = []
        
        # Pattern matching approach
        for i in range(self.env.physics.model.njnt):
            joint_name = self.env.physics.model.id2name(i, 'joint')
            if joint_name:
                for pattern in right_arm_patterns:
                    if pattern in joint_name.lower():
                        identified_joints.append(joint_name)
                        break
        
        # Position-based fallback approach
        if not identified_joints:
            try:
                torso_id = self.env.physics.model.name2id('torso', 'body')
                if torso_id != -1:
                    for i in range(self.env.physics.model.njnt):
                        if self.env.physics.model.jnt_bodyid[i] == torso_id:
                            joint_name = self.env.physics.model.id2name(i, 'joint')
                            if joint_name and ('right' in joint_name.lower() or '_r' in joint_name.lower()):
                                identified_joints.append(joint_name)
            except:
                pass
        
        # Generic fallback if no joints found
        if not identified_joints:
            print("Warning: Could not identify right arm joints by name. Using generic indices.")
            identified_joints = ['generic_right_shoulder', 'generic_right_elbow']
            
        return identified_joints

    def _get_right_arm_joint_ids(self):
        """Convert right arm joint names to model joint indices."""
        joint_ids = []
        
        for joint_name in self.right_arm_joints:
            if joint_name.startswith('generic_'):
                # Use estimated indices for generic joints
                if joint_name == 'generic_right_shoulder':
                    joint_ids.append(10)
                elif joint_name == 'generic_right_elbow':
                    joint_ids.append(11)
            else:
                # Get actual index for named joints
                try:
                    joint_id = self.env.physics.model.name2id(joint_name, 'joint')
                    if joint_id != -1:
                        joint_ids.append(joint_id)
                except:
                    print(f"Warning: Joint {joint_name} not found in model.")
        
        return joint_ids

    def _normalize_obs(self, obs):
        """Normalize observations using running statistics."""
        if self.obs_running_mean is None:
            self.obs_running_mean = np.zeros_like(obs)
            self.obs_running_var = np.ones_like(obs)
            return obs
        
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
        
        return (obs - self.obs_running_mean) / (np.sqrt(self.obs_running_var) + self.obs_epsilon)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            if hasattr(super(), 'reset'):
                super().reset(seed=seed)
            if hasattr(self.env._task, 'random'):
                self.env._task.random.seed(seed)

        time_step = self.env.reset()

        # Apply custom standing position
        with self.env.physics.reset_context():
            qpos = self.env.physics.data.qpos.copy()
            qvel = self.env.physics.data.qvel.copy()
            self._apply_standing_position(qpos, qvel, randomize=self.init_randomization)
            self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset state variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self._get_height()
        self.total_reward = 0.0
        self.lying_down_steps = 0
        self.prev_height = self._get_height()
        self.last_action = None
        self.was_falling = False
        self.wave_time = 0.0
        self.standing_time = 0

        # Process observation
        obs = self._flatten_obs(time_step.observation)
        norm_obs = self._normalize_obs(obs)
        foot_heights = self._get_foot_heights()
        
        info = {
            'height': self._get_height(),
            'foot_heights': foot_heights,
            'right_arm_joints': self.right_arm_joints,
            'right_arm_joint_ids': self.right_arm_joint_ids
        }

        return norm_obs.astype(np.float32), info
    
    def _apply_standing_position(self, qpos, qvel, randomize=0.0):
        """Set the humanoid in a stable standing position."""
        # Set height
        target_height = 1.35
        qpos[2] = target_height
        
        # Set orientation
        if randomize > 0.0:
            # Add slight randomization for robustness
            w = 1.0 - randomize * 0.1 * np.random.random()
            x = randomize * 0.1 * (np.random.random() - 0.5)
            y = randomize * 0.1 * (np.random.random() - 0.5)
            z = randomize * 0.2 * (np.random.random() - 0.5)
            
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            qpos[3:7] = [w/norm, x/norm, y/norm, z/norm]
        else:
            qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Perfect upright
        
        # Set joint positions
        joint_indices = slice(7, len(qpos))
        qpos[joint_indices] = 0.0  # Zero all joints initially
        
        try:
            # Configure key joints for stability
            knee_indices = [self.env.physics.model.name2id('left_knee', 'joint'),
                          self.env.physics.model.name2id('right_knee', 'joint')]
            knee_indices = [i-7 for i in knee_indices if i != -1]
            
            # Find hip joints
            hip_indices = []
            for joint_name in ['left_hip_x', 'left_hip_y', 'left_hip_z', 
                              'right_hip_x', 'right_hip_y', 'right_hip_z']:
                try:
                    idx = self.env.physics.model.name2id(joint_name, 'joint')
                    if idx != -1:
                        hip_indices.append(idx-7)
                except:
                    pass
            
            # Find ankle joints
            ankle_indices = []
            for joint_name in ['left_ankle', 'right_ankle']:
                try:
                    idx = self.env.physics.model.name2id(joint_name, 'joint')
                    if idx != -1:
                        ankle_indices.append(idx-7)
                except:
                    pass
            
            # Set knee bend
            for idx in knee_indices:
                if idx >= 0:
                    base_knee_angle = 0.1
                    if randomize > 0.0:
                        knee_random = randomize * 0.05 * np.random.random()
                        qpos[joint_indices][idx] = base_knee_angle + knee_random
                    else:
                        qpos[joint_indices][idx] = base_knee_angle
            
            # Set hip flexion
            for idx in hip_indices:
                if idx >= 0:
                    if 'hip_y' in self.env.physics.model.id2name(idx+7, 'joint'):
                        base_hip_angle = 0.05
                        if randomize > 0.0:
                            hip_random = randomize * 0.03 * (np.random.random() - 0.5)
                            qpos[joint_indices][idx] = base_hip_angle + hip_random
                        else:
                            qpos[joint_indices][idx] = base_hip_angle
            
            # Set ankle flexion
            for idx in ankle_indices:
                if idx >= 0:
                    base_ankle_angle = -0.05
                    if randomize > 0.0:
                        ankle_random = randomize * 0.03 * (np.random.random() - 0.5)
                        qpos[joint_indices][idx] = base_ankle_angle + ankle_random
                    else:
                        qpos[joint_indices][idx] = base_ankle_angle
            
            # Add diverse pose randomization
            if randomize > 0.1:
                all_joints = slice(7, len(qpos))
                qpos[all_joints] += randomize * 0.1 * np.random.randn(len(qpos[all_joints]))
        
        except Exception as e:
            pass
        
        # Set velocities
        if randomize > 0.0:
            qvel[:] = randomize * 0.01 * np.random.randn(*qvel.shape)
        else:
            qvel[:] = 0.0

    def step(self, action):
        """Take a step through the environment."""
        action = action.astype(np.float64)
        
        # Apply action smoothing
        if self.last_action is not None and self.action_smoothing > 0:
            action = self.action_smoothing * self.last_action + (1 - self.action_smoothing) * action
        
        self.last_action = action.copy()
        self.wave_time += self.env.physics.timestep()
        
        # Execute physics step
        time_step = self.env.step(action)
        self.steps_this_episode += 1
        
        # Process observation
        obs_flat = self._flatten_obs(time_step.observation).astype(np.float32)
        norm_obs = self._normalize_obs(obs_flat)
        
        # Track state
        self.was_falling = self.lying_down_steps > 5
        current_height = self._get_height()
        
        # Update standing time
        if current_height > 0.8:
            self.standing_time += 1
        else:
            self.standing_time = 0
        
        # Calculate reward components
        stand_reward = self._compute_stand_reward()
        wave_reward = self._compute_wave_reward()
        time_reward = self._compute_time_reward()
        
        # Recovery bonus
        recovery_reward = 0
        if self.was_falling and current_height > 0.8:
            recovery_reward = 5.0
        
        # Total reward
        reward = stand_reward + wave_reward + time_reward + recovery_reward
        self.total_reward += reward
        
        # Termination conditions
        terminated = time_step.last()
        foot_heights = self._get_foot_heights()
        
        # Check for falling
        if current_height < 0.3:
            self.lying_down_steps += 1
        else:
            self.lying_down_steps = 0
            
        # Early termination
        fall_terminated = self.lying_down_steps >= self.lying_down_threshold
        terminated = terminated or fall_terminated
        truncated = self.steps_this_episode >= self.max_steps
        
        # Track max height
        if current_height > self.best_height_this_episode:
            self.best_height_this_episode = current_height
        self.prev_height = current_height
            
        # Prepare info
        info = {
            'height': current_height,
            'foot_heights': foot_heights,
            'steps': self.steps_this_episode,
            'fall_terminated': fall_terminated,
            'jump_detected': any(h > 0.2 for h in foot_heights),
            'total_reward': self.total_reward,
            'best_height': self.best_height_this_episode,
            'recovery_reward': recovery_reward,
            'wave_reward': wave_reward,
            'stand_reward': stand_reward,
            'time_reward': time_reward,
            'standing_time': self.standing_time
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
                'standing_time': self.standing_time
            }
        
        return norm_obs, float(reward), terminated, truncated, info

    def _get_height(self):
        """Get current torso height."""
        if hasattr(self.env.physics, 'torso_height'):
            return self.env.physics.torso_height()
        return self.env.physics.named.data.geom_xpos['torso', 'z']
    
    def _get_foot_heights(self):
        """Get the heights of both feet."""
        try:
            left_foot_height = self.env.physics.named.data.geom_xpos['left_foot', 'z']
            right_foot_height = self.env.physics.named.data.geom_xpos['right_foot', 'z']
            return [float(left_foot_height), float(right_foot_height)]
        except:
            return [0.1, 0.1]
    
    def _get_right_arm_state(self):
        """Get positions and velocities of right arm joints."""
        arm_positions = []
        arm_velocities = []
        
        for joint_id in self.right_arm_joint_ids:
            try:
                pos = self.env.physics.data.qpos[joint_id]
                arm_positions.append(pos)
                
                vel = self.env.physics.data.qvel[joint_id - 1]
                arm_velocities.append(vel)
            except:
                arm_positions.append(0.0)
                arm_velocities.append(0.0)
        
        return arm_positions, arm_velocities
    
    def _compute_wave_reward(self):
        """Calculate reward for arm waving behavior."""
        # Target sinusoidal pattern
        target_wave = self.wave_amplitude * np.sin(2 * np.pi * self.wave_frequency * self.wave_time)
        
        # Get current arm state
        arm_positions, arm_velocities = self._get_right_arm_state()
        
        if not arm_positions:
            return 0.0
        
        wave_reward = 0.0
        
        # Position matching component
        if len(arm_positions) > 0:
            position_diff = abs(arm_positions[0] - target_wave)
            position_match = max(0.0, 1.0 - position_diff / self.wave_amplitude)
            wave_reward += position_match
        
        # Velocity alignment component
        if len(arm_velocities) > 0:
            target_velocity = self.wave_amplitude * 2 * np.pi * self.wave_frequency * np.cos(2 * np.pi * self.wave_frequency * self.wave_time)
            velocity_alignment = np.sign(arm_velocities[0]) == np.sign(target_velocity)
            if velocity_alignment:
                wave_reward += 0.5
        
        return float(wave_reward * self.wave_reward_weight)
    
    def _compute_time_reward(self):
        """Calculate time-based standing reward."""
        time_reward = self.time_reward_scale * self.standing_time
        max_time_reward = 2.0
        return float(min(time_reward, max_time_reward))
            
    def _compute_stand_reward(self):
        """Calculate standing reward based on posture and stability."""
        physics = self.env.physics
        
        # Height reward
        try:
            head_height = physics.named.data.geom_xpos['head', 'z']
        except:
            head_height = self._get_height()
        
        STAND_HEIGHT = 1.4
        
        if head_height < 0.05:
            height_reward = -3.0
        else:
            norm_height = (head_height - 0.05) / (STAND_HEIGHT - 0.05)
            norm_height = max(0.0, min(1.0, norm_height))
            height_reward = 4.0 * (norm_height - 0.5)
        
        # Orientation reward
        try:
            quat = physics.data.qpos[3:7]
            upright_reward = quat[0] ** 2
        except:
            upright_reward = 0.5
        
        # Foot position reward
        foot_heights = self._get_foot_heights()
        foot_ground_contact = sum(1.0 for h in foot_heights if h < 0.1)
        foot_reward = 0.5 * (foot_ground_contact / len(foot_heights))
        
        # Stability reward
        try:
            ang_vel = physics.data.qvel[3:6]
            ang_vel_norm = np.linalg.norm(ang_vel)
            stability_reward = -min(1.0, ang_vel_norm / 5.0)
        except:
            stability_reward = 0.0
            
        # Energy efficiency
        try:
            energy_penalty = -0.1 * np.sum(np.square(physics.data.ctrl))
        except:
            energy_penalty = 0.0
        
        # Height stability
        height_stability_reward = 0.0
        if self.prev_height is not None:
            height_velocity = abs(self._get_height() - self.prev_height) / physics.timestep()
            height_stability_reward = -0.3 * min(1.0, height_velocity)
            
        # Zero moment point stability
        try:
            com_pos = physics.center_of_mass_position()
            com_vel = physics.center_of_mass_velocity()
            gravity = 9.81
            
            foot_positions = []
            try:
                foot_positions.append(physics.named.data.geom_xpos['left_foot'])
                foot_positions.append(physics.named.data.geom_xpos['right_foot'])
            except:
                foot_positions = [[0, 0, 0], [0, 0, 0]]
                
            foot_center = np.mean(foot_positions, axis=0)
            
            zmp_x = com_pos[0] + com_vel[0] * np.sqrt(com_pos[2] / gravity)
            zmp_y = com_pos[1] + com_vel[1] * np.sqrt(com_pos[2] / gravity)
            
            zmp_dist = np.sqrt((zmp_x - foot_center[0])**2 + (zmp_y - foot_center[1])**2)
            zmp_reward = -0.5 * min(1.0, zmp_dist)
        except:
            zmp_reward = 0.0
        
        # Combined reward
        combined_reward = (
            2.0 * height_reward +
            1.5 * upright_reward +
            1.0 * foot_reward +
            0.5 * stability_reward +
            energy_penalty +
            height_stability_reward +
            zmp_reward
        )
        
        return float(combined_reward)

    def _flatten_obs(self, obs_dict):
        """Convert dictionary observation to flat vector."""
        obs_list = []
        for key in self._obs_keys:
            val = obs_dict[key]
            if isinstance(val, (int, float, np.number)):
                obs_list.append(np.array([val], dtype=np.float64))
            else:
                obs_list.append(np.asarray(val, dtype=np.float64).flatten())
        return np.concatenate(obs_list)

    def render(self, mode='human', height=480, width=640, camera_id=None):
        """Render the environment state."""
        if camera_id is None:
            camera_id = 0 if mode == 'human' else 2
            
        if mode == 'rgb_array':
            return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        elif mode == 'human':
            return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Clean up environment resources."""
        if hasattr(self.env, 'close'):
            self.env.close()
