
from agent import Agent
import numpy as np
import argparse


class World():
    """
    World holds state data.
    """

    def __init__(self, behavior_size=3, time=10):
        if behavior_size < 1:
            print("behavior size must be 1 or more; will be set to 3 (default).")
            self.behavior = 3
        else:
            self.behavior_size = int(behavior_size)
        if time < 1:
            print("time must be 1 or more; will be set to 10 (default).")
            self.time = 10
        else:
            self.time = int(time)
        self.agents = []
        self.b_priors = []
        self.behaviors = []
        self.predictions = []
        self.errors = []
        self.costs = []

    def create_agents(self):
        '''
        Generate the agents.
        '''
        # later on we can add more agents
        num_agents = 2
        for i in range(num_agents):
            self.agents.append(Agent(self.behavior_size))

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
                p = self.agents[i].get_priors()
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
                    self.agents[i].get_world(self.behaviors[-1][1])
                else:
                    self.agents[i].get_world(self.behaviors[-1][0])
                p = self.agents[i].make_prediction()
                e = self.agents[i].behavior_prediction_error()
                self.agents[i].learn_conform()
                self.agents[i].learn_predict_world()
                c = self.agents[i].get_cost()
                prediction.append(p)
                error.append(e)
                cost.append(c)
            self.predictions.append(prediction)
            self.errors.append(error)
            self.costs.append(cost)
            time_left -= 1

    def get_results(self):
        '''
        Print the results of the experiment.
        '''
        for i in range(self.time):
            print("time step:   {}".format(i+1))
            print("bhv priors:  {}".format(
                [b[0].tolist() for b in self.b_priors[i]]))
            print("behaviors:   {}".format(
                [b[0].tolist() for b in self.behaviors[i]]))
            print("predictions: {}".format(
                [b[0].tolist() for b in np.round(self.predictions[i], 3)]))
            print("errors:      {}".format(self.errors[i]))
            print("net costs:   {}".format(np.round(self.costs[i], 3)))
            print("  ---  ")


def main():
    parser = argparse.ArgumentParser(description="A world generator")
    '''parser.add_argument("-n", "--num_agents", type=int,
                        metavar="num_agents", help="number of agents in the experiment")
    '''
    parser.add_argument("-b", "--behavior_size", type=int,
                        metavar="behavior_size", help="size of behavior vector")
    parser.add_argument("-t", "--time", type=int,
                        metavar="time", help="number of time steps")

    args = parser.parse_args()
    world = World(args.behavior_size, args.time)
    world.create_agents()
    world.run()
    world.get_results()


if __name__ == '__main__':
    """
    The main function called when world.py is run
    from the command line:

    > ./world.py -b 3 -t 10

    See the usage string for more details.

    > python ./world.py --help
    """
    main()
