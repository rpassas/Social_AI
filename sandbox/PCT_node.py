from .Effector import Effector


class PCT_node:

    def __init__(self, state_size, sensor, comparator, Kp, Ki, Kd, control_update):
        assert state_size > 0, "state size must be > 0"
        self.state_size = state_size
        self.sensor = sensor
        self.comparator = comparator
        self.effector = Effector(Kp, Ki, Kd)
        self.control_update = control_update

    def sense(self, observation):
        sensory_signal = self.sensor(observation)
        return sensory_signal

    def compare(self, reference_signal, sensory_signal):
        error = self.comparator(reference_signal, sensory_signal)
        return error

    def effect(self, error):
        output = self.effector.get_output(error)
        return output

    def update_control(self, error):
        Kp, Ki, Kd = self.effector.get_params()
        Kp_new, Ki_new, Kd_new = self.control_update(error, Kp, Ki, Kd)
        self.effector.set_params(Kp_new, Ki_new, Kd_new)

    def go(self, reference_signal, observation):
        sense = self.sense(observation)
        error = self.compare(reference_signal, sense)
        output = self.effect(error)
        self.update_control(error)
        return output
