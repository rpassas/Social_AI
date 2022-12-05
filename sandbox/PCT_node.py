from .Effector import Effector


class PCT_node:

    def __init__(self,
                 sensor,
                 comparator,
                 generate_reference,
                 control_update,
                 Kp=0,
                 Kd=0,
                 Ki=0,
                 parents=[],
                 output_limits=(None, None)):
        self.sensor = sensor
        self.comparator = comparator
        self.effector = Effector(Kp, Ki, Kd)
        self.generate_reference = generate_reference
        self.control_update = control_update
        self.output_limits = output_limits
        self.parents = parents
        self.sensory_signal = []
        self.error = []
        self.reference = []
        self.output = []

    def sense(self, observation):
        self.sensory_signal = self.sensor(observation)
        return self.sensory_signal

    def compare(self, reference_signal, sensory_signal):
        if len(reference_signal) != sensory_signal:
            raise ValueError("Sensory signal must match reference signal.")
        self.error = self.comparator(reference_signal, sensory_signal)
        return self.error

    def effect(self, error):
        self.output = self.effector.get_output(error)
        return self.output

    def update_control(self, error):
        Kp, Ki, Kd = self.effector.get_params()
        Kp_new, Ki_new, Kd_new = self.control_update(error, Kp, Ki, Kd)
        Ki_new = self.bound(Ki_new)  # prevent windup
        self.effector.set_params(Kp_new, Ki_new, Kd_new)

    def generate_reference(self):
        if not self.parents:
            self.reference = self.reference_update(self.reference, self.error)
        else:
            inputs = [p.get_output() for p in self.parents]
            self.reference = self.generate_reference(inputs, self.error)

    def go(self, observation):
        reference_signal = self.generate_reference()
        sense = self.sense(observation)
        error = self.compare(reference_signal, sense)
        output = self.effect(self.error)
        self.update_control(error)
        output = self.bound(output)
        self.reference = reference_signal
        self.output = output
        return output

    def get_output(self):
        return self.output

    def get_error(self):
        return self.error

    def bound(self, val):
        lower = min(self.output_limits)
        upper = max(self.output_limits)
        if upper is not None and val > upper:
            return upper
        if lower is not None and val < lower:
            return lower
        return val
