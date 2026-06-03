"""
DWA planner with acceleration + steering-angle sampling, JAX-accelerated,
terrain-aware (roll/pitch from a 2D elevation grid) and payload-sway-aware
(single-pendulum dynamics for a suspended load).

Date:   17/12/2024

State convention (vehicle): [x, y, theta, v, delta]
    x, y   : position in the planning frame [m]
    theta  : heading [rad]
    v      : forward velocity [m/s]
    delta  : steering angle [rad]

State convention (payload): [theta_y, theta_z, thetadot_y, thetadot_z]
    pendulum angles + angular rates about y and z axes.

Control: [a, delta_target]
    a            : acceleration command [m/s^2]
    delta_target : target steering angle [rad] (applied directly each step)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from collision_check import Occupy_map
from single_pend import SinglePendulum


class DWAplanner_sway:

    def __init__(self,
                 veh_configs,
                 robot_init_location,
                 terrain,
                 terrain_origin,
                 terrain_resolution,
                 obstacles,
                 rope_length=3.0,
                 payload_mass=500.0,
                 sway_weight=1.2):
        """
        Args:
            veh_configs:         AMC() instance with kinematic + sampling parameters.
            robot_init_location: [x, y, theta] of the planning-frame origin in the
                                 terrain frame. Used to rotate/translate planned
                                 trajectories before querying the elevation grid.
            terrain:             jnp.ndarray (Nrows, Ncols), 2D elevation grid [m].
            terrain_origin:      tuple (x_min, y_min) — world coordinates of the
                                 grid cell (0, 0) [m].
            terrain_resolution:  float [m/cell] — grid resolution.
            obstacles:           jnp.ndarray (N, 2) — point-obstacle coordinates.
            rope_length:         payload rope length [m] for the pendulum model.
            payload_mass:        payload mass [kg] for the pendulum model.
            sway_weight:         weight of the payload-sway cost term (set 0 to
                                 disable sway suppression).
        """
        self.Configs = veh_configs
        self.robot_init_location = robot_init_location

        # Terrain
        self.terrain = terrain
        self.terrain_shape = self.terrain.shape
        self.terrain_origin = jnp.asarray(terrain_origin, dtype=jnp.float32)
        self.terrain_resolution = float(terrain_resolution)

        # Obstacles + collision map (footprint length uses the wheelbase L).
        self.obstacles = jnp.asarray(obstacles, dtype=jnp.float32)
        L, W = veh_configs.L, veh_configs.robot_width
        self.occmap = Occupy_map(W, L, 0.0, self.obstacles)

        # Payload dynamics (derived once via SymPy → JAX lambdas)
        _pendulum = SinglePendulum(rope_length, payload_mass)
        self.q_freelamda = _pendulum.cal_eom(IF_JAX=True)

        # Cost weighting
        self.sway_weight = float(sway_weight)

        # State set by pick_best_action — read by cal_cost via closure
        self.if_last_point = False
        self.distance_to_final = 0.0

    @partial(jit, static_argnums=(0,))
    def pick_best_action(self, veh_state, payload_state, target_point, obstacles,
                         IF_last_point=False, distance_to_final=0.0):
        """
        Evaluate the dynamic window and return the best (lowest-cost, collision-free)
        trajectory.

        Args:
            veh_state:         (5,) [x, y, theta, v, delta]
            payload_state:     (4,) [theta_y, theta_z, thetadot_y, thetadot_z]
            target_point:      (2,) [x_goal, y_goal]
            obstacles:         (N, 2) obstacle coordinates (can change between calls)
            IF_last_point:     True when this waypoint is the final goal — enables
                               gradual slowdown.
            distance_to_final: distance from current state to the final goal [m];
                               used for the slowdown velocity profile.

        Returns:
            best_traj:    (N+1, 5) selected trajectory (state at each prediction step)
            trajectories: (M, N+1, 5) all candidate trajectories for plotting/debug
        """
        self.if_last_point = IF_last_point
        self.distance_to_final = distance_to_final

        a_dotw_mesh_grid = self.cal_dynamic_window(veh_state)
        trajectories, costs = vmap(
            self.cal_trajectories, in_axes=(0, None, None, None, None)
        )(a_dotw_mesh_grid, veh_state, payload_state, target_point, obstacles)
        min_idx = jnp.argmin(costs)
        best_traj = trajectories[min_idx]
        return best_traj, trajectories

    @partial(jit, static_argnums=(0,))
    def cal_dynamic_window(self, current_state):
        """Sample (acceleration, steering_angle) on a regular grid."""
        cfg = self.Configs
        a_samples = jnp.linspace(-cfg.max_acceleration, cfg.max_acceleration,
                                 cfg.num_v_samples + 1)
        delta_samples = jnp.linspace(-cfg.max_steering_angle, cfg.max_steering_angle,
                                     cfg.num_w_samples + 1)
        A, Delta = jnp.meshgrid(a_samples, delta_samples)
        return jnp.stack([A.ravel(), Delta.ravel()], axis=1)

    @partial(jit, static_argnums=(0,))
    def cal_trajectories(self, sample_cmd, state, payload_curr_states,
                         target_point, obstacles):
        """Roll the bicycle model forward for `prediction_time` and score the result."""
        v_sample, delta_sample = sample_cmd  # NB: v_sample holds the sampled acceleration
        N = int(self.Configs.prediction_time / self.Configs.dt)

        def step_fn(curr_state, _):
            cmd = jnp.array([v_sample, delta_sample], dtype=jnp.float32)
            next_state = self._motion_model(curr_state, cmd)
            return next_state, next_state

        _, states = lax.scan(step_fn, state, None, length=N)
        trajectory = jnp.vstack([state[None, :], states])
        cost = self.cal_cost(trajectory, payload_curr_states, target_point, obstacles)
        return trajectory, cost

    @partial(jit, static_argnums=(0,))
    def _motion_model(self, state: jnp.ndarray, cmd: jnp.ndarray) -> jnp.ndarray:
        """Bicycle kinematics: v <- v + a*dt (clipped), delta <- delta_target (clipped)."""
        a_cmd, delta_target = cmd
        x, y, theta, v, _ = state
        cfg = self.Configs

        v_new = jnp.clip(v + a_cmd * cfg.dt, 0.0, cfg.max_speed)
        delta_new = jnp.clip(delta_target, -cfg.max_steering_angle, cfg.max_steering_angle)

        x += v_new * jnp.cos(theta) * cfg.dt
        y += v_new * jnp.sin(theta) * cfg.dt
        theta += (v_new / cfg.L) * jnp.tan(delta_new) * cfg.dt
        return jnp.array([x, y, theta, v_new, delta_new], dtype=jnp.float32)

    @partial(jit, static_argnums=(0,))
    def cal_cost(self,
                 trajectory: jnp.ndarray,
                 payload_curr_states: jnp.ndarray,
                 target_point: jnp.ndarray,
                 obstacles: jnp.ndarray) -> jnp.ndarray:
        """
        Cost of a single trajectory:
            +inf if collision (in the terrain frame); else
            v_cost + 3 * heading_cost + sway_weight * sway_cost + 0.5 * steering_change_cost.

        The first step is to rotate/translate the planning-frame trajectory into the
        terrain frame using `robot_init_location`, so the elevation lookup and
        obstacle check are both done in terrain coordinates.
        """
        xs, ys, thetas = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

        rot = self.robot_init_location[-1]
        c, s = jnp.cos(rot), jnp.sin(rot)
        x_rot = xs * c - ys * s + self.robot_init_location[0]
        y_rot = xs * s + ys * c + self.robot_init_location[1]
        yaw_rot = thetas + rot

        traj_rot3 = jnp.stack([x_rot, y_rot, yaw_rot], axis=1)
        x_end, y_end, theta_end = xs[-1], ys[-1], thetas[-1]

        is_collision = self.occmap.check_collision_trajectory(traj_rot3, obstacles)

        def compute_noncoll(_):
            # Sway cost via single-pendulum dynamics driven by boom-tip acceleration
            roll, pitch, z_rot = self.get_roll_pitch(x_rot, y_rot, yaw_rot)
            sway_max = self.cal_sway_cost(
                x_rot, y_rot, z_rot, yaw_rot, roll, pitch, payload_curr_states
            )
            sway_cost = sway_max / (jnp.pi / 4)  # normalise to ~[0, 1]

            # Heading cost in [0, 1]
            dx = target_point[0] - x_end
            dy = target_point[1] - y_end
            raw_diff = jnp.arctan2(dy, dx) - theta_end
            raw_diff = (raw_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
            heading_cost = jnp.abs(raw_diff) / jnp.pi

            # Velocity cost: gradual slowdown near the final goal
            slow_start_distance = 15.0
            min_distance = 1.5
            ratio = jnp.clip(
                (self.distance_to_final - min_distance) / (slow_start_distance - min_distance),
                0.0, 1.0,
            )
            target_velocity = lax.cond(
                self.if_last_point,
                lambda _: self.Configs.max_speed * ratio,
                lambda _: self.Configs.max_speed,
                operand=None,
            )
            v_cost = jnp.abs(trajectory[-1, 3] - target_velocity) / self.Configs.max_speed

            # Steering smoothness: penalise the jump from the current steering angle
            # to the commanded one.
            steering_jump = jnp.abs(trajectory[1, 4] - trajectory[0, 4])
            steering_change_cost = steering_jump / self.Configs.max_steering_angle
            steering_cost = 0.5 * steering_change_cost

            # Cost-term weights below are illustrative demo defaults (sway_weight configurable).
            return v_cost + 3.0 * heading_cost + self.sway_weight * sway_cost + steering_cost

        cost = lax.cond(
            is_collision,
            lambda _: jnp.array(float('inf'), dtype=jnp.float32),
            compute_noncoll,
            operand=0,
        )
        return cost

    @partial(jit, static_argnums=(0,))
    def cal_sway_cost(self, x, y, z, theta, roll, pitch, payload_curr_states):
        """Predict the maximum payload sway over the planning horizon."""
        # Boom-tip world coordinates
        x_boom, y_boom, z_boom = self.compute_boom_displacement(
            4.0, 0.0, 7.0,                       # boom geometry: forward 4 m, up 7 m
            theta.flatten(), pitch.flatten(), roll.flatten(),
        )
        x_boom += x; y_boom += y; z_boom += z

        dt = self.Configs.dt
        x_dot = jnp.gradient(x_boom, dt)
        y_dot = jnp.gradient(y_boom, dt)
        z_dot = jnp.gradient(z_boom, dt)
        y_ddot = -jnp.gradient(x_dot, dt)
        x_ddot = jnp.gradient(y_dot, dt)
        z_ddot = jnp.gradient(z_dot, dt)

        payload_traj = self.ode_ana_rk4_jax(
            self.q_freelamda, x_ddot, y_ddot, z_ddot, payload_curr_states
        )
        return jnp.max(jnp.abs(payload_traj[:, 0])) + jnp.max(jnp.abs(payload_traj[:, 1]))

    @partial(jit, static_argnums=(0,))
    def get_roll_pitch(self, x, y, yaw):
        """
        Estimate roll, pitch, and average elevation at each timestep by sampling
        the elevation grid at the four wheel positions.

        Returns:
            roll, pitch, elevation — each shape (T,)
        """
        cfg = self.Configs
        L, W = cfg.L, cfg.robot_width

        offsets = jnp.array([
            [ L / 2,  W / 2],  # Front Left
            [ L / 2, -W / 2],  # Front Right
            [-L / 2,  W / 2],  # Rear Left
            [-L / 2, -W / 2],  # Rear Right
        ])
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        off_x = offsets[:, 0][None, :]
        off_y = offsets[:, 1][None, :]
        rot_x = c[:, None] * off_x - s[:, None] * off_y
        rot_y = s[:, None] * off_x + c[:, None] * off_y
        rotated = jnp.stack([rot_x, rot_y], axis=-1)              # (T, 4, 2)
        base = jnp.stack([x, y], axis=1)[:, None, :]              # (T, 1, 2)
        wheel_pos = base + rotated                                 # (T, 4, 2)
        Z = jax.vmap(lambda pts: jax.vmap(self._query_elevation)(pts))(wheel_pos)

        Z_FL, Z_FR, Z_RL, Z_RR = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
        Z_front = (Z_FL + Z_FR) / 2.0
        Z_rear  = (Z_RL + Z_RR) / 2.0
        Z_left  = (Z_FL + Z_RL) / 2.0
        Z_right = (Z_FR + Z_RR) / 2.0

        # Small-angle terrain-tilt estimate from the four wheel heights.
        pitch = jnp.arcsin((Z_front - Z_rear) / L)
        roll  = jnp.arcsin((Z_left  - Z_right) / W)
        elevation = (Z_FL + Z_FR + Z_RL + Z_RR) / 4.0
        return roll.flatten(), pitch.flatten(), elevation.flatten()

    @partial(jit, static_argnums=(0,))
    def get_min_cost_trajectory(self, trajectory, cost):
        """Convenience helper: pick the argmin candidate."""
        min_idx = jnp.argmin(cost)
        return trajectory[min_idx], cost[min_idx]

    @partial(jit, static_argnums=(0,))
    def _query_elevation(self, positions):
        """
        Nearest-cell elevation lookup.

        Args:
            positions: jnp.ndarray (..., 2) — [x, y] in the terrain frame [m].

        Returns:
            jnp.ndarray of elevations — one value per input position.
        """
        n_rows, n_cols = self.terrain_shape

        # Shift to grid origin, scale by resolution, floor to integer cells.
        shifted = positions - self.terrain_origin[None, :]
        idx_f = shifted / self.terrain_resolution
        idx_i = jnp.floor(idx_f).astype(jnp.int32)

        row_idx = jnp.clip(idx_i[:, 1], 0, n_rows - 1)
        col_idx = jnp.clip(idx_i[:, 0], 0, n_cols - 1)

        flat_terrain = self.terrain.reshape(-1)
        linear_idx = row_idx * n_cols + col_idx
        return flat_terrain[linear_idx]

    @partial(jit, static_argnums=(0, 1))
    def ode_ana_rk4_jax(self, q_freelamda, x_ddot, y_ddot, z_ddot, paylaod_inista):
        """
        Integrate the single-pendulum payload ODE with RK4 over the prediction horizon.

        Args:
            q_freelamda:   (yq_fn, zq_fn) callables returning theta_y_ddot and theta_z_ddot.
            x_ddot/y_ddot/z_ddot: shape (N,) — driving accelerations of the boom tip.
            paylaod_inista: (4,) initial payload state.

        Returns:
            traj: (N, 4) integrated pendulum state.
        """
        h = self.Configs.dt
        N = int(self.Configs.prediction_time / h)
        yq_fn, zq_fn = q_freelamda

        def get_accel(time):
            # Clamp the lookup index to the prediction horizon.
            idx = jnp.minimum((time / h).astype(jnp.int32), N - 1)
            return jnp.take(x_ddot, idx), jnp.take(y_ddot, idx), jnp.take(z_ddot, idx)

        def deriv(time, y):
            θy, θz, θy_dot, θz_dot = y
            xdd, ydd, zdd = get_accel(time)
            θy_dd = yq_fn(xdd, ydd, zdd, θy, θy_dot, θz, θz_dot)
            θz_dd = zq_fn(xdd, ydd, zdd, θy, θy_dot, θz, θz_dot)
            return jnp.stack([θy_dot, θz_dot, θy_dd, θz_dd])

        def scan_body(carry, _):
            time, state = carry
            new_state = self.rk4_step(deriv, time, state, h)
            return (time + h, new_state), new_state

        _, traj = lax.scan(scan_body, (0.0, paylaod_inista), None, length=N)
        return traj

    @staticmethod
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h,     y + h * k3)
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    @jit
    def compute_boom_displacement(delta_x, delta_y, delta_z,
                                  yaw_arr, pitch_arr, roll_arr):
        """
        Boom-tip displacement in the world frame for an offset (delta_x, delta_y, delta_z)
        in the vehicle body frame, given yaw/pitch/roll arrays (radians).
        """
        cy, sy = jnp.cos(yaw_arr),   jnp.sin(yaw_arr)
        cp, sp = jnp.cos(pitch_arr), jnp.sin(pitch_arr)
        cr, sr = jnp.cos(roll_arr),  jnp.sin(roll_arr)

        # Rz * Ry * Rx applied to (delta_x, delta_y, delta_z)
        boom_x = cy * cp * delta_x + (cy * sp * sr - sy * cr) * delta_y + (cy * sp * cr + sy * sr) * delta_z
        boom_y = sy * cp * delta_x + (sy * sp * sr + cy * cr) * delta_y + (sy * sp * cr - cy * sr) * delta_z
        boom_z = -sp        * delta_x +  cp * sr               * delta_y +  cp * cr               * delta_z
        return boom_x, boom_y, boom_z
