# Terrain-mobile-crane
A JAX-accelerated motion planner for an Autonomous Mobile Crane (AMC) carrying a suspended payload. The planner
samples *(acceleration, steering-angle)* pairs, rolls the bicycle model forward
in parallel with `jax.vmap` + `jax.lax.scan`, and scores each candidate
trajectory with a cost that combines:

- **Velocity** tracking (with gradual slowdown near the goal),
- **Heading** to the goal,
- **Collision** check against point obstacles (returns `+∞`),
- **Payload sway**, predicted from a single-pendulum model driven by the
  boom-tip acceleration that results from rolling over the 2D elevation
  grid at each candidate's wheel positions, and
- **Steering smoothness**.

The whole evaluation pipeline is JIT-compiled, so all candidate trajectories are
scored in a single vectorised XLA program.

## State conventions

- **Vehicle state** `[x, y, theta, v, delta]` — position, heading, velocity, steering angle.
- **Payload state** `[theta_y, theta_z, thetadot_y, thetadot_z]` — pendulum angles + rates.
- **Control** `[a, delta_target]` — acceleration command + target steering angle.

## Citation

```bibtex
@article{zhou2026terrain,
  title={Terrain-adaptive motion planning for autonomous mobile cranes with load sway suppression},
  author={Zhou, Zhuomin and Bai, Yu and Abdi, Elahe},
  journal={Automation in Construction},
  volume={187},
  pages={106964},
  year={2026},
  publisher={Elsevier}
}
```

This repository is a simplified, illustrative release of the core algorithm described in the paper.