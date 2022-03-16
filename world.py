
from agent_of_chaos import Agent_of_Chaos
from agent_average import Agent_Average
from agent_dummy import Agent_Dummy
import numpy as np
import argparse


class World():
    """
    World holds state data.
    """

    def __init__(self, behavior_size=3, time=15, agent=["chaos", "chaos"], alphas=[], betas=[], n=2):
        # argparse will make unfilled optional args 'None', so perform checks
        # behavior size
        if not behavior_size:
            self.behavior_size = 3
        elif behavior_size < 1:
            print("behavior size must be 1 or more; will be set to 3 (default).")
            self.behavior_size = 3
        else:
            self.behavior_size = int(behavior_size)
        # length of an experiment
        if not time:
            self.time = 15
        elif time < 1:
            print("time must be 1 or more; will be set to 10 (default).")
            self.time = 15
        else:
            self.time = int(time)
        # agent types
        if not agent:
            self.type = ["chaos", "chaos"]
        else:
            self.type = agent
        # state dimension
        if not n:
            self.n = 3
        elif n < 1 or n > 25:
            self.n = 3
        else:
            self.n = n
        # alpha is the conformity learning rate
        if alphas:
            if len(alphas) < self.n:
                self.alphas = [0.5]*self.n
            else:
                self.alphas = alphas
        else:
            self.alphas = [0.5]*self.n
        # beta is the prediction learning rate
        if betas:
            if len(betas) < self.n:
                self.betas = [0.5]*self.n
            else:
                self.betas = betas
        else:
            self.betas = [0.5]*self.n
        # variables to be filled as the experiment runs
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
        n = self.n
        while n:
            n -= 1
            if self.type[n-1] == "chaos":
                self.agents.append(Agent_of_Chaos(
                    self.behavior_size, float(self.alphas[n-1]), float(self.betas[n-1])))
            elif self.type[n-1] == "average":
                self.agents.append(Agent_Average(
                    self.behavior_size, float(self.alphas[n-1]), float(self.betas[n-1])))
            else:
                self.agents.append(Agent_Dummy(
                    self.behavior_size, float(self.alphas[n-1]), float(self.betas[n-1])))

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

    def get_priors(self):
        '''
        Get a representation of the priors so each list is an agents priors across time.
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
            print("agent alpha:   {}".format(a.get_alpha()))
            print("agent beta:   {}".format(a.get_beta()))
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
    parser = argparse.ArgumentParser(description="A world generator")
    '''parser.add_argument("-n", "--num_agents", type=int,
                        metavar="num_agents", help="number of agents in the experiment")
    '''
    parser.add_argument("-s", "--behavior_size", type=int,
                        metavar="behavior_size", help="size of behavior vector")
    parser.add_argument("-t", "--time", type=int,
                        metavar="time", help="number of time steps")
    parser.add_argument("-q", "--agent", type=str, nargs='+',
                        metavar="agent", help="type of agents to be used: chaos, average, dummy, static")
    parser.add_argument("-a", "--alpha", type=float, nargs='+',
                        metavar="alpha", help="prior learning rate: 0.001 - 1")
    parser.add_argument("-b", "--beta", type=float, nargs='+',
                        metavar="beta", help="conformity learning rate: 0.001 - 1")
    parser.add_argument("-r", "--seed", type=int,
                        metavar="seed", help="random seed for generating priors")

    args = parser.parse_args()

    world = World(args.behavior_size, args.time,
                  args.agent, args.alpha, args.beta, args.seed)
    world.create_agents()
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
