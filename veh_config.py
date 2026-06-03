"""
Vehicle configuration for the AMC (Autonomous Mobile Crane).

The values below are illustrative defaults for the demo and may differ from the
calibrated parameters reported in the paper.
"""


class AMC:
    def __init__(self):
        # Kinematics
        self.L = 3.6                       # wheelbase [m]
        self.robot_width = 2.0             # vehicle width [m]
        self.robot_length = 5.0            # vehicle length (for collision footprint) [m]

        # Planner timing
        self.dt = 0.05                     # control / prediction step [s]
        self.prediction_time = 3.0         # planning horizon [s]

        # Sampling resolution
        self.num_v_samples = 30            # acceleration samples
        self.num_w_samples = 30            # steering angle samples

        # Limits
        self.max_acceleration = 3.0        # [m/s^2]
        self.max_steering_angle = 0.6      # [rad]
        self.max_steering_angle_rate = 0.6 # [rad/s]
        self.max_speed = 2.78               # [m/s]
        self.min_speed = 0.0               # [m/s]
