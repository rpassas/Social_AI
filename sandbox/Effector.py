class Effector:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def get_output(self, error):
        derivative = error - self.prev_error
        self.prev_error = error
        self.integral = self.integral + error
        out = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        return out

    def get_params(self):
        return self.Kp, self.Ki, self.Kd

    def set_params(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
