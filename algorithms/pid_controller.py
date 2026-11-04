
"""
PID Controller for Adaptive IoT Downsampling

Input:
    Measured MQTT delivery delay (milliseconds)

Output:
    Updated sensor data retention percentage

Higher network delay -> Lower retention
Lower network delay  -> Higher retention

This module can be tested independently before
integration with the MQTT producer.
"""

from math import isfinite


class PIDRetentionController:

    def __init__(
        self,
        target_delay_ms=100.0,
        initial_keep=0.20,
        min_keep=0.05,
        max_keep=0.50,
        kp=0.0015,
        ki=0.00003,
        kd=0.00001,
        smoothing=0.4,
    ):

        if not (
            0.0 < min_keep <= initial_keep <= max_keep <= 1.0
        ):
            raise ValueError(
                "Require 0 < min_keep <= initial_keep "
                "<= max_keep <= 1."
            )

        if not isfinite(target_delay_ms) or target_delay_ms <= 0:
            raise ValueError(
                "Target delay must be finite and positive."
            )

        if not 0.0 < smoothing <= 1.0:
            raise ValueError(
                "Smoothing must be between 0 and 1."
            )

        gains = (kp, ki, kd)

        if any(
            not isfinite(gain) or gain < 0
            for gain in gains
        ):
            raise ValueError(
                "PID gains must be finite and nonnegative."
            )

        self.target_delay_ms = float(target_delay_ms)

        self.initial_keep = float(initial_keep)
        self.min_keep = float(min_keep)
        self.max_keep = float(max_keep)

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.smoothing = float(smoothing)

        self.keep = self.initial_keep

        self.smoothed_delay_ms = None
        self.integral = 0.0
        self.previous_error = None

    # ========================================================
    # UPDATE PID CONTROLLER
    # ========================================================

    def update(self, measured_delay_ms, dt_seconds):

        delay = float(measured_delay_ms)
        dt = float(dt_seconds)

        if not isfinite(delay) or delay < 0:
            raise ValueError(
                "Measured delay must be finite and nonnegative."
            )

        if not isfinite(dt) or dt <= 0:
            raise ValueError(
                "Elapsed time must be finite and positive."
            )

        # Smooth fluctuations in network measurements.
        if self.smoothed_delay_ms is None:

            self.smoothed_delay_ms = delay

        else:

            self.smoothed_delay_ms = (
                self.smoothing * delay
                + (1.0 - self.smoothing)
                * self.smoothed_delay_ms
            )

        # Positive error means we can increase retention.
        # Negative error means we should reduce retention.
        error = (
            self.target_delay_ms
            - self.smoothed_delay_ms
        )

        # Derivative term.
        if self.previous_error is None:

            derivative = 0.0

        else:

            derivative = (
                error - self.previous_error
            ) / dt

        # Candidate integral.
        candidate_integral = (
            self.integral + error * dt
        )

        def calculate_output(integral_value):

            return (
                self.initial_keep
                + self.kp * error
                + self.ki * integral_value
                + self.kd * derivative
            )

        proposed = calculate_output(
            candidate_integral
        )

        # Anti-windup:
        # Avoid accumulating error beyond controller limits.
        pushing_above_max = (
            proposed > self.max_keep
            and error > 0
        )

        pushing_below_min = (
            proposed < self.min_keep
            and error < 0
        )

        if not (
            pushing_above_max or pushing_below_min
        ):

            self.integral = candidate_integral

        proposed = calculate_output(
            self.integral
        )

        # Enforce retention boundaries.
        self.keep = max(
            self.min_keep,
            min(self.max_keep, proposed)
        )

        self.previous_error = error

        return self.keep

    # ========================================================
    # CURRENT CONTROLLER STATUS
    # ========================================================

    def status(self):

        return {
            "target_delay_ms": self.target_delay_ms,
            "smoothed_delay_ms": self.smoothed_delay_ms,
            "retention": self.keep,
            "data_reduction_pct": (
                1.0 - self.keep
            ) * 100.0,
        }


# ============================================================
# TEST PID CONTROLLER WITHOUT MQTT
# ============================================================

if __name__ == "__main__":

    controller = PIDRetentionController()

    print("\nPID CONTROLLER SIMULATION")
    print("=" * 65)

    print(
        f"{'Scenario':<20}"
        f"{'Delay':<15}"
        f"{'Retention':<15}"
        f"{'Reduction'}"
    )

    scenarios = (
        [("Normal", 30.0)] * 3
        + [("Congested", 250.0)] * 6
        + [("Recovered", 30.0)] * 9
    )

    for scenario, measured_delay in scenarios:

        retention = controller.update(
            measured_delay_ms=measured_delay,
            dt_seconds=1.0
        )

        reduction = (
            1.0 - retention
        ) * 100.0

        print(
            f"{scenario:<20}"
            f"{measured_delay:<15.1f}"
            f"{retention * 100:<15.1f}"
            f"{reduction:.1f}%"
        )

    print("=" * 65)
    print("PID simulation completed.")