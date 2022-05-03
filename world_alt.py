import numpy as np


class World():
    """
    World holds state data.
    INPUTS:
        time [integer, default=100]: sets number of experimental trials, t.
        agent [agents]: list of agents 
        seed [integer, default=None]: use an integer seed in order to replicate experiments.
    """

    def __init__(self, time=100, seed=None, agents=[]):
        if seed:
            np.random.seed(seed)
        assert time > 0, "time must be > 0"  # length of an experiment
        self.time = time
        # variables to be filled as the experiment runs
        self.agents = []
        self.b_priors = []
        self.behaviors = []
        self.predictions = []
        self.errors = []
        self.costs = []

    def run(self):
        '''
        Run experiment and record results.
        '''
        time_left = self.time
        while time_left:
            # generate behaviors
            prior = []
            behavior = []
            for i in range(len(self.agents)):
                b = self.agents[i].make_behavior()
                p = self.agents[i].get_behav_priors()
                behavior.append(b)
                prior.append(p)
                # print("behavior of agent {}: ".format(i) + str(behavior))
            self.behaviors.append(behavior)
            self.b_priors.append(prior)

            # receive behaviors, predict, learn
            # will have to be updated for multi agents
            prediction = []
            error = []
            cost = []
            for i in range(len(self.agents)):
                if i == 0:
                    # agent 0 gets agent 1's behavior
                    self.agents[i].get_world(self.behaviors[-1][1])
                else:
                    # agent 1 gets agent 0's behavior
                    self.agents[i].get_world(self.behaviors[-1][0])
                p = self.agents[i].make_prediction()
                dif, avg_abs_error = self.agents[i].behavior_prediction_error()
                self.agents[i].learn_conform()
                self.agents[i].learn_predict_world()
                # c = self.agents[i].get_cost()
                c = avg_abs_error
                prediction.append(p)
                error.append(dif)
                cost.append(c)
            self.predictions.append(prediction)
            self.errors.append(error)
            self.costs.append(cost)
            time_left -= 1

    def get_agents(self):
        '''
        Get agent types.
        '''
        agents = [a.get_type() for a in self.agents]
        return agents

    def get_errors(self):
        '''
        Get a representation of the errors so each list is an agents error across time.
        '''
        error_array = np.array(self.errors)
        error_T = error_array.T
        return error_T

    def get_costs(self):
        '''
        Get a representation of the costs so each list is an agents cost across time (cumulative error).
        '''
        cost_array = np.array(self.costs)
        cost_T = cost_array.T
        return cost_T

    def get_pred(self):
        '''
        Get a representation of the predictions so each list is an agents prediction across time.
        '''
        pred_array = np.array(self.predictions)
        pred_T = pred_array.T
        return pred_T

    def get_behav_priors(self):
        '''
        Get a representation of the priors so each list is an agent's behavioral priors across time.
        '''
        prior_array = np.array(self.b_priors)
        prior_T = prior_array.T
        return prior_T

    def print_results(self):
        '''
        Print the results of the experiment.
        '''
        for a in self.agents:
            print("agent type:   {}".format(a.get_type()))
            print("  ---  ")

        print("\n")
        for i in range(self.time):
            print("time step:   {}".format(i+1))
            print("bhv priors:  {}".format(
                [b.tolist() for b in np.round(self.b_priors[i], 3)]))
            print("behaviors:   {}".format(
                [b.tolist() for b in self.behaviors[i]]))
            print("predictions: {}".format(
                [b.tolist() for b in np.round(self.predictions[i], 3)]))
            print("errors:      {}".format(self.errors[i]))
            print("net costs:   {}".format(np.round(self.costs[i], 3)))
            print("  ---  ")


def main():
    world = World()
    world.run()
    world.print_results()


if __name__ == '__main__':
    """
    The main function called when world.py is run
    from the command line:
    > python world.py
    See the usage string for more details.
    > python ./world.py --help
    """
    main()
