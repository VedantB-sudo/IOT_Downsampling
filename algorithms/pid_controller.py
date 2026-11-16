class PID:
    def __init__(self):
        self.retention = 0.30
        self.integral = 0
        self.previous_error = 0

    def update(self, queue):
        error = 0.25 - queue
        derivative = error - self.previous_error

        new_integral = max(-2, min(2, self.integral + error))
        trial = 0.30 + 0.55 * error + 0.05 * new_integral + 0.10 * derivative

        # Prevent integral windup
        if 0.05 <= trial <= 0.50 or (trial > 0.50 and error < 0) or (trial < 0.05 and error > 0):
            self.integral = new_integral

        target = 0.30 + 0.55 * error + 0.05 * self.integral + 0.10 * derivative
        change = max(-0.08, min(0.08, target - self.retention))

        self.retention = max(0.05, min(0.50, self.retention + change))
        self.previous_error = error
        return self.retention


# Simulated network conditions
schedule = ([("Normal", 90)] * 4 +
            [("Congested", 12)] * 8 +
            [("Recovered", 90)] * 10)

pid = PID()
queue = drops = total_selected = 0

print(f"{'Scenario':<14} {'Queue':<10} {'Retention':<12} {'Reduction'}")
print("-" * 52)

for scenario, capacity in schedule:
    occupancy = queue / 240
    retention = pid.update(occupancy)
    selected = round(200 * retention)
    total_selected += selected

    queue = max(0, queue + selected - capacity)
    drops += max(0, queue - 240)
    queue = min(queue, 240)

    print(f"{scenario:<14} {occupancy:<10.1%} {retention:<12.1%} {1-retention:.1%}")


# Fixed retention using the same total sample budget
fixed_queue = fixed_drops = 0
base, remainder = divmod(total_selected, len(schedule))

for i, (_, capacity) in enumerate(schedule):
    selected = base + ((i + 1) * remainder // len(schedule)
                       - i * remainder // len(schedule))

    fixed_queue = max(0, fixed_queue + selected - capacity)
    fixed_drops += max(0, fixed_queue - 240)
    fixed_queue = min(fixed_queue, 240)

print("\nRESULTS")
print(f"Average retention : {total_selected / (len(schedule) * 200):.1%}")
print(f"PID queue drops   : {drops}")
print(f"Fixed queue drops : {fixed_drops}")