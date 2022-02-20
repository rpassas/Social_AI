
from agent import Agent
import argparse


class World():
    """
    World holds state data.
    """

    def __init__(self, behavior=3, time=10):
        if behavior < 1:
            print("behavior size must be 1 or more; will be set to 3 (default).")
            self.behavior = 3
        else:
            self.behavior = behavior
        if self.time < 1:
            print("time must be 1 or more; will be set to 10 (default).")
            self.time = 10
        else:
            self.time = time
        self.agents = []
        self.behaviors = []
        self.predictions = []
        self.errors = []
        self.accuracies = []
        self.costs = []

    def create_agents(self):
        # later on we can add more agents
        num_agents = 2
        for i in range(num_agents):
            self.agents.append(Agent(self.behavior))

    def run(self):
        time_left = self.time
        while time_left:
            # generate behaviors
            behavior = []
            for i in range(len(self.agents)):
                b = self.agents[i].make_behavior()
                b.append(behavior)
                #print("behavior of agent {}: ".format(i) + str(behavior))
            self.behaviors.append(behavior)

            # receive behaviors, predict, learn
            # will have to be updated for multi agents
            prediction = []
            error = []
            accuracy = []
            cost = []
            for i in range(len(self.agents)):
                if i == 0:
                    self.agents[i].get_world(self.behaviors[1])
                else:
                    self.agents[i].get_world(self.behaviors[0])
                p = self.agents[i].make_prediction()
                e = self.agents[i].behavior_prediction_error()
                self.agents[i].learn_conform()
                self.agents[i].learn_predict_world()
                a = self.agents[i].get_acc()
                c = self.agets[i].get_cost()
                p.append(prediction)
                e.append(error)
                a.append(accuracy)
                c.append(cost)
            self.predictions.append(prediction)
            self.errors.append(error)
            self.accuracies.append(accuracy)
            self.costs.append(cost)
            time_left -= 1

        def get_results(self):

            for i in range(self.time):
                print("time step:   {}".format(i))
                print("behaviors:   {}".format(self.behaviors[i]))
                print("predictions: {}".format(self.predictions[i]))
                print("errors:      {}".format(self.errors[i]))
                print("accuracies:  {}".format(self.accuracies[i]))
                print("net costs:   {}".format(self.costss[i]))


def main():
    parser = argparse.ArgumentParser(description="A world generator")
    '''parser.add_argument("-n", "--num_agents", type=int,
                        metavar="num_agents", help="number of agents in the experiment")
    '''
    parser.add_argument("-b", "--behavior", type=int,
                        metavar="behavior", help="size of behavior vector")
    parser.add_argument("-t", "--time", type=int,
                        metavar="time", help="number of time steps")

    args = parser.parse_args()
    world = World(args.behavior, args.time)
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
