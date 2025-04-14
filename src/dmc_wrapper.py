import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_control.rl import control # Added for base Environment class check
import collections # Added for OrderedDict check

class DMCWrapper(gym.Env):
    """Wrapper for dm_control environments to make them compatible with Gymnasium.
    Enhanced with curriculum learning and reward shaping for better standing and waving."""

    def __init__(self, domain_name="humanoid", task_name="stand",
                 enable_waving=False, initial_standing_assist=0.8,
                 assist_decay_rate=0.9999, max_steps=1000, seed=None): # Added seed, decay_rate

        # Load the environment
        # Use a fixed random state for the environment for reproducibility within the wrapper instance if seed provided
        random_state = np.random.RandomState(seed) if seed is not None else None
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': random_state})

        # Get observation specs and determine total dimensions
        obs_spec = self.env.observation_spec()
        if not isinstance(obs_spec, collections.OrderedDict):
             print("Warning: Observation spec is not an OrderedDict. Flattening order might be inconsistent.")
             # Convert to OrderedDict if possible, otherwise use default dict order (less reliable)
             obs_spec = collections.OrderedDict(obs_spec)

        self._obs_keys = list(obs_spec.keys()) # Store keys for consistent flattening
        total_obs_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())

        # Define the observation space (continuous vector of all observations combined)
        # Use float64 for DMC compatibility, then convert to float32 later if needed by SB3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float64
        )

        # Get action specs
        action_spec = self.env.action_spec()

        # Define the action space
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32 # SB3 typically expects float32 actions
        )

        # Flag to enable/disable waving behavior
        self.enable_waving = enable_waving

        # Curriculum learning parameters
        self.initial_standing_assist = initial_standing_assist
        self.standing_assist_decay = assist_decay_rate # Use passed decay rate
        self.current_standing_assist = initial_standing_assist

        # Episode tracking
        self.max_steps = max_steps
        self.steps_this_episode = 0
        self.best_height_this_episode = 0.0
        self.no_progress_steps = 0
        self.total_stand_reward = 0.0
        self.total_wave_reward = 0.0


        # Waving-related variables (initialize only if needed)
        if self.enable_waving:
            self._init_wave_vars()
            # Identify right arm joints (adjust if your model differs)
            # Example indices for standard humanoid: shoulder_pitch, shoulder_roll, elbow_pitch
            self.right_arm_joint_indices = [
                self.env.physics.model.name2id('right_shoulder_pitch', 'joint'),
                self.env.physics.model.name2id('right_shoulder_roll', 'joint'),
                self.env.physics.model.name2id('right_elbow_pitch', 'joint')
            ]
            # Filter out invalid indices (-1)
            self.right_arm_joint_indices = [idx for idx in self.right_arm_joint_indices if idx != -1]
            if not self.right_arm_joint_indices:
                 print("Warning: Could not find right arm joint indices by name. Waving reward will be disabled.")
                 self.enable_waving = False # Disable if joints not found


        # Metadata for rendering (optional)
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': int(1.0 / self.env.physics.timestep())}


    def _init_wave_vars(self):
        """Initialize variables needed for wave reward calculation."""
        self.prev_arm_positions = None
        self.prev_arm_velocities = None # Store previous velocities as well
        self.prev_shoulder_delta = 0.0
        self.wave_cycle_points = 0 # Track points within a potential wave cycle
        self.direction_changes = 0


    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        # Seed the environment's random state if a seed is provided
        # Note: This seeds the environment's internal randomness (if any in task),
        # not the physics engine directly in the same way suite.load does.
        if seed is not None:
             super().reset(seed=seed)
             # We might want to re-seed the task's random state more directly if possible,
             # but often suite.load handles the primary seeding.
             if hasattr(self.env._task, 'random'):
                 self.env._task.random.seed(seed)

        # Reset the dm_control environment
        time_step = self.env.reset()

        # --- Apply standing assistance (Curriculum) ---
        # Decay is applied *after* the episode completes (in step),
        # so here we just use the current_standing_assist value.
        if self.current_standing_assist > 0.01: # Apply assist if it's still significant
            with self.env.physics.reset_context():
                qpos = self.env.physics.data.qpos.copy()
                qvel = self.env.physics.data.qvel.copy()

                # 1. Raise initial height (z-position, index 2)
                target_height = 1.2 + 0.2 * np.random.uniform(-0.5, 0.5) # Base height + small noise
                qpos[2] = target_height * self.current_standing_assist + qpos[2] * (1 - self.current_standing_assist)

                # 2. Set torso orientation closer to upright (indices 3-6 are quaternion w,x,y,z)
                # Target: [1, 0, 0, 0] with noise scaled by (1 - assist)
                noise = (np.random.rand(3) - 0.5) * 0.3 * (1 - self.current_standing_assist)
                target_quat = np.array([1.0, 0.0, 0.0, 0.0])
                # Apply noise rotation (simplified approach) - better would be SLERP or axis-angle
                qpos[4:7] += noise # Add noise to x,y,z components
                qpos[3:7] /= np.linalg.norm(qpos[3:7]) # Re-normalize quaternion

                # 3. Nudge joint angles towards a neutral standing pose (optional, can be complex)
                # Instead of forcing specific angles, let's just add small noise scaled by assist level
                joint_indices = slice(7, len(qpos)) # Assuming joints start at index 7
                qpos[joint_indices] += np.random.normal(0, 0.1 * (1 - self.current_standing_assist), size=qpos[joint_indices].shape)

                # 4. Set initial velocities to near zero for stability
                qvel[:] *= (1 - self.current_standing_assist) # Dampen existing velocities based on assist
                qvel[:] += np.random.normal(0, 0.05 * (1-self.current_standing_assist), size=qvel.shape) # Add small noise

                # Apply the modified state
                self.env.physics.set_state(np.concatenate([qpos, qvel]))

        # Reset episode-specific variables
        self.steps_this_episode = 0
        self.best_height_this_episode = self.env.physics.torso_height() if hasattr(self.env.physics, 'torso_height') else self.env.physics.named.data.geom_xpos['torso', 'z']
        self.no_progress_steps = 0
        self.total_stand_reward = 0.0
        self.total_wave_reward = 0.0

        if self.enable_waving:
            self._init_wave_vars() # Reset wave variables

        # Get initial observation and info
        obs = self._flatten_obs(time_step.observation)
        info = {'standing_assist': self.current_standing_assist} # Initial info

        # Ensure obs is float32 for SB3
        return obs.astype(np.float32), info


    def step(self, action):
        """Take a step in the environment."""
        # Ensure action is float64 for dm_control if necessary, but usually float32 is fine
        action = action.astype(np.float64)

        # Execute action in the dm_control environment
        time_step = self.env.step(action)

        # Increment step counter
        self.steps_this_episode += 1

        # Get flattened observation
        obs_dict = time_step.observation
        obs_flat = self._flatten_obs(obs_dict).astype(np.float32) # Return float32

        # --- Calculate Rewards ---
        # Use physics state directly for reward calculation where needed
        physics_state = self.env.physics

        # 1. Standing Reward
        stand_reward = self._compute_stand_reward(obs_dict, physics_state)
        self.total_stand_reward += stand_reward

        # 2. Waving Reward (if enabled and standing)
        wave_reward = 0.0
        current_height = physics_state.torso_height() if hasattr(physics_state, 'torso_height') else physics_state.named.data.geom_xpos['torso', 'z']
        is_standing_enough = current_height > 1.1 # Threshold to enable waving reward

        if self.enable_waving and is_standing_enough:
            wave_reward = self._compute_wave_reward(obs_dict, physics_state)
             # Apply wave reward progressively based on height and stability? Maybe just weight it.
            wave_weight = 0.2 # Adjust this weight based on importance relative to standing
            total_reward = stand_reward + wave_weight * wave_reward
            self.total_wave_reward += wave_reward # Track unweighted wave reward
        else:
            total_reward = stand_reward

        # --- Check Termination and Truncation ---
        # Termination: Environment signals end (e.g., fell down in dm_control stand task)
        terminated = time_step.last() # dm_control uses time_step.last() for termination

        # Truncation: Max steps reached or early termination due to lack of progress
        truncated = False
        early_termination_reason = "none"

        # Early termination logic
        if current_height > self.best_height_this_episode + 0.01: # Require some minimum improvement
            self.best_height_this_episode = current_height
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # Terminate if no height progress for a while AND still low
        if self.no_progress_steps > 250 and current_height < 0.8:
             truncated = True # Treat as truncation (task failed)
             early_termination_reason = "no_progress_low_height"

        # Truncate if max steps reached
        if self.steps_this_episode >= self.max_steps:
            truncated = True
            early_termination_reason = "max_steps_reached"


        # --- Update Curriculum ---
        # Decay standing assistance IF the episode ended naturally (terminated or truncated)
        if terminated or truncated:
            self.current_standing_assist = max(0.0, self.current_standing_assist * self.standing_assist_decay)


        # --- Prepare Info Dictionary ---
        info = {
            'stand_reward_step': stand_reward, # Reward for this step
            'wave_reward_step': wave_reward,
            'height': current_height,
            'steps': self.steps_this_episode,
            'standing_assist': self.current_standing_assist, # Current assist level
            'early_termination_reason': early_termination_reason,
            'is_standing_enough': is_standing_enough, # For debugging wave activation
        }
        # Add final info when episode ends (SB3 uses this)
        if terminated or truncated:
            info['final_info'] = { # SB3 expects episode summary here
                'stand_reward': self.total_stand_reward,
                'wave_reward': self.total_wave_reward, # Log total unweighted wave reward
                 'height': current_height, # Final height
                 'steps': self.steps_this_episode,
                 'standing_assist': self.current_standing_assist, # Assist level at end of ep
                 'early_termination_reason': early_termination_reason,
                 'max_height_episode': self.best_height_this_episode,
                 'TimeLimit.truncated': truncated and not terminated, # Indicate if truncated by time limit specifically
                 'terminal_observation': obs_flat, # SB3 uses this
            }
            # Add legacy keys if needed by ProgressCallback structure
            info.update(info['final_info'])


        # Return (obs, reward, terminated, truncated, info)
        return obs_flat, total_reward, terminated, truncated, info


    def _flatten_obs(self, obs_dict):
        """Flatten the observation dictionary into a 1D numpy array using stored keys."""
        obs_list = []
        for key in self._obs_keys:
            val = obs_dict[key]
            if isinstance(val, (int, float, np.number)):
                 obs_list.append(np.array([val], dtype=np.float64))
            else:
                 obs_list.append(np.asarray(val, dtype=np.float64).flatten())
        return np.concatenate(obs_list)


    def _compute_stand_reward(self, observation, physics_state):
        """Enhanced reward shaping for standing, using physics state."""

        # 1. Base reward from DM Control task (often checks height, uprightness)
        # Accessing the private _task is generally discouraged, but sometimes necessary for the raw reward.
        # Alternative: Replicate the core reward logic here if known.
        if hasattr(self.env, '_task') and hasattr(self.env._task, 'get_reward'):
             base_reward = float(self.env._task.get_reward(physics_state))
        else:
             # Fallback/alternative: Reward based on height
             current_height = physics_state.torso_height() if hasattr(physics_state, 'torso_height') else physics_state.named.data.geom_xpos['torso', 'z']
             base_reward = max(0.0, min(current_height / 1.4, 1.0)) # Simple height reward scaled 0-1


        # 2. Height Reward Component
        current_height = physics_state.torso_height() if hasattr(physics_state, 'torso_height') else physics_state.named.data.geom_xpos['torso', 'z']
        # Encourage reaching target height (e.g., ~1.4m), penalize being too low
        height_reward = 1.0 * np.exp(-5.0 * (current_height - 1.4)**2) # Gaussian centered at 1.4m
        # Add a small constant reward just for being above ground
        height_reward += 0.1 if current_height > 0.5 else -0.5


        # 3. Upright Orientation Reward
        # Use torso orientation (quaternion or z-axis of rotation matrix)
        # physics.torso_upright() is simple if available, physics.torso_orientation() gives quat
        if hasattr(physics_state, 'torso_upright'):
             upright_value = physics_state.torso_upright() # Ranges ~0 (down) to 1 (up)
             upright_reward = 0.5 * upright_value
        else:
             # Use quaternion (w, x, y, z) - index 3-6 in qpos
             # A simple check is the 'w' component (qpos[3]) or z-component of transformed z-axis
             quat = physics_state.data.qpos[3:7]
             # Project world z-axis [0,0,1] by inverse quaternion rotation to get body's z-axis in world frame
             # Simpler: Check 'w' component's deviation from 1 (upright) or -1 (upside down)
             # Even simpler: Check z-component of head/torso geom position relative to feet/pelvis
             # Using torso geom z-axis from rotation matrix is more robust:
             torso_z_axis = physics_state.named.data.xmat['torso'][6:9] # z-axis vector (3rd column)
             upright_value = max(0.0, torso_z_axis[2]) # z-component of the body's z-axis (should be close to 1 when upright)
             upright_reward = 0.5 * upright_value


        # 4. Stability Reward (Penalize Velocity)
        # Linear velocity of torso
        lin_vel_norm = np.linalg.norm(physics_state.named.data.sensordata['torso_vel'][0:3])
        # Angular velocity of torso
        ang_vel_norm = np.linalg.norm(physics_state.named.data.sensordata['torso_vel'][3:6])
        # Joint velocities (indices 6+ in qvel)
        joint_vel = physics_state.data.qvel[6:] # Assuming first 6 are free joint
        joint_vel_norm = np.linalg.norm(joint_vel)

        # Penalize excessive linear/angular velocity and joint velocity
        velocity_penalty = 0.1 * lin_vel_norm + 0.05 * ang_vel_norm + 0.02 * joint_vel_norm
        stability_reward = -min(velocity_penalty, 1.0) # Cap penalty


        # 5. Control Effort Penalty (Optional but good)
        control_effort = np.sum(np.square(physics_state.data.ctrl))
        control_penalty = -0.001 * min(control_effort, 10.0) # Small penalty for excessive control signals


        # --- Combine Rewards ---
        # Adjust weights based on importance
        # Let's try: base (if good), height, upright, stability
        total_shaped_reward = (
            base_reward * 0.5 +       # Weight the original reward
            height_reward * 1.0 +     # Strong incentive for height
            upright_reward * 0.8 +    # Strong incentive for uprightness
            stability_reward * 1.0 +  # Penalize instability
            control_penalty * 1.0     # Penalize high controls
        )

        # Scale reward based on curriculum progress (more reward as task gets harder)
        # curriculum_scale = 1.0 + 1.0 * max(0.0, (self.initial_standing_assist - self.current_standing_assist))
        # Simpler: just return the shaped reward. Let the agent figure it out.
        # Scaling can sometimes obscure the learning signal.

        return total_shaped_reward


    def _compute_wave_reward(self, observation, physics_state):
        """Compute reward for wave-like motion of the right arm."""
        if not self.enable_waving or not self.right_arm_joint_indices:
            return 0.0

        # Get current arm joint angles and velocities
        qpos = physics_state.data.qpos
        qvel = physics_state.data.qvel
        joint_indices = slice(7, len(qpos)) # Assuming joints start at index 7
        joint_vel_indices = slice(6, len(qvel)) # Assuming joint vels start at index 6

        # Map global joint indices to local indices within the joint vectors
        # This requires knowing the full joint list order. A simpler way is to use named access if available.
        try:
             # Use named access if possible (more robust)
             r_shoulder_pitch_vel = physics_state.named.data.qvel['right_shoulder_pitch']
             r_shoulder_pitch_pos = physics_state.named.data.qpos['right_shoulder_pitch']
             r_shoulder_roll_pos = physics_state.named.data.qpos['right_shoulder_roll']
             # Add elbow etc. if needed
        except KeyError:
             # Fallback to indices (less robust) - requires knowing exact index mapping
             print("Warning: Using hardcoded indices for wave reward. May fail if joint order changes.")
             # Find the indices within the joint slice
             base_joint_qpos_idx = 7
             base_joint_qvel_idx = 6
             local_arm_indices_pos = [idx - base_joint_qpos_idx for idx in self.right_arm_joint_indices]
             local_arm_indices_vel = [idx - base_joint_qvel_idx for idx in self.right_arm_joint_indices]

             if any(i < 0 for i in local_arm_indices_pos + local_arm_indices_vel):
                  print("Error: Invalid joint index mapping for waving.")
                  return 0.0 # Cannot calculate reward

             arm_positions = qpos[joint_indices][local_arm_indices_pos]
             arm_velocities = qvel[joint_vel_indices][local_arm_indices_vel]
             r_shoulder_pitch_vel = arm_velocities[0] # Assuming pitch is first in self.right_arm_joint_indices
             r_shoulder_pitch_pos = arm_positions[0]
             r_shoulder_roll_pos = arm_positions[1] # Assuming roll is second


        wave_reward = 0.0

        # --- Wave criteria ---
        # 1. Arm Elevation (Shoulder Pitch): Reward raising the arm (negative angle is up/forward)
        target_pitch = -1.0 # Target angle for raised arm (radians)
        pitch_reward = 0.5 * np.exp(-3.0 * (r_shoulder_pitch_pos - target_pitch)**2) # Gaussian around target
        wave_reward += pitch_reward

        # 2. Arm Out to Side (Shoulder Roll): Reward rolling arm outwards (negative angle)
        target_roll = -0.5
        roll_reward = 0.3 * np.exp(-4.0 * (r_shoulder_roll_pos - target_roll)**2)
        wave_reward += roll_reward


        # 3. Waving Motion (Velocity Changes): Reward changes in shoulder pitch velocity when arm is raised
        if r_shoulder_pitch_pos < -0.5: # Only reward waving if arm is somewhat raised
             current_delta_vel = r_shoulder_pitch_vel # Use velocity directly

             # Check for direction change (crossing zero velocity with sufficient magnitude)
             if self.prev_arm_velocities is not None:
                 prev_shoulder_pitch_vel = self.prev_arm_velocities[0]
                 vel_threshold = 0.1 # Min velocity magnitude to count as movement
                 if np.sign(current_delta_vel) != np.sign(prev_shoulder_pitch_vel) and \
                    abs(current_delta_vel) > vel_threshold and abs(prev_shoulder_pitch_vel) > vel_threshold:
                     self.direction_changes += 1
                     wave_reward += 0.5 # Reward for direction change

             # Reward moderate velocity
             velocity_magnitude = abs(r_shoulder_pitch_vel)
             ideal_velocity = 0.8
             velocity_reward = 0.2 * np.exp(-5.0 * (velocity_magnitude - ideal_velocity)**2)
             wave_reward += velocity_reward


        # Store current state for next step
        # Need to handle potential key errors if using named access fallback
        try:
            self.prev_arm_positions = np.array([r_shoulder_pitch_pos, r_shoulder_roll_pos]) # Store relevant positions
            self.prev_arm_velocities = np.array([r_shoulder_pitch_vel]) # Store relevant velocities
        except NameError: # Handles case where variables weren't assigned due to key error
            pass

        # Bonus for sustained waving (multiple direction changes)
        wave_reward += 0.1 * min(self.direction_changes, 5)

        return wave_reward


    def render(self, mode='human', height=480, width=640, camera_id=None):
        """Render the environment."""
        if camera_id is None:
             # Default camera ID, often 0 (fixed) or 2 (tracking 'egocentric') for humanoids
             camera_id = 0 if mode == 'human' else 2 # Use tracking camera for rgb_array

        if mode == 'rgb_array':
            return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        elif mode == 'human':
             # dm_control viewer runs externally, render call might just update it
             return self.env.physics.render(height=height, width=width, camera_id=camera_id)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Close the environment."""
        # dm_control environments don't typically have an explicit close method in the suite wrapper
        # but the underlying physics engine might. Usually handled by garbage collection.
        if hasattr(self.env, 'close'):
             self.env.close()
