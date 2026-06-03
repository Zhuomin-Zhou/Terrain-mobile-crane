"""
Vectorised collision checking for the DWA planner.

The robot footprint is approximated by a circle of radius
    sqrt((W/2)^2 + (L/2)^2) + margin
so a single squared-distance comparison per (trajectory_point, obstacle) pair
is enough to detect collision.
"""

import jax.numpy as jnp


class Occupy_map:
    def __init__(self, robot_width, robot_length, margin, obstacles):
        """
        Args:
            robot_width:  float [m]
            robot_length: float [m]
            margin:       float, additional safety distance [m]
            obstacles:    jnp.ndarray (N, 2), [x, y] coordinates of obstacle points
        """
        self.width = robot_width
        self.length = robot_length
        self.margin = margin

        self.obstacles = jnp.array(obstacles)

        # Conservative circular approximation: robot fits inside a circle of
        # radius equal to half the body diagonal, plus the safety margin.
        self.robot_radius = jnp.sqrt((robot_width / 2) ** 2 + (robot_length / 2) ** 2) + margin
        self.safe_dist_sq = self.robot_radius ** 2

    def check_collision_trajectory(self, trajectory, obstacles):
        """
        Full pairwise collision check.

        Args:
            trajectory: jnp.ndarray (T, 3) — [x, y, theta] along the rollout
            obstacles:  jnp.ndarray (N, 2)

        Returns:
            jnp.bool_ — True if any (trajectory point, obstacle) pair is within
                       the safety radius.
        """
        traj_pos = trajectory[:, :2]
        diff = traj_pos[:, None, :] - obstacles[None, :, :]  # (T, N, 2)
        dist_sq = jnp.sum(diff ** 2, axis=-1)                # (T, N)
        return jnp.any(dist_sq < self.safe_dist_sq)
