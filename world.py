
from agent_of_chaos import Agent_of_Chaos
from agent_average_behavior import Agent_Average
from agent_average_prediction import Agent_Average_Prediction
from agent_dummy import Agent_Dummy
from agent_with_model import Agent_with_Model
from agent_with_sigmoid_model import Agent_with_Sigmoid_Model
from agent_with_linear_model import Agent_with_Linear_Model
import numpy as np
import argparse


class World():
    """
    World holds state data.

    INPUTS:
        state_size [integer, default=3]: sets size of behavior feature space, N.
        time [integer, default=100]: sets number of experimental trials, t.
        agent
        alphas
        betas
        seed [integer, default=None]: use an integer seed in order to replicate analyses.
        memory
        agent_n [integer, default=2]: sets number of agents. Currently only set-up to handle 2.

    """

    def __init__(self, state_size=3, time=100, agent=["model_sig", "model_sig"],
        seed=None, memory=[4, 4], behav_control = [4, 4], agent_n=2):
        if seed:
            np.random.seed(seed)
        # argparse will make unfilled optional args 'None', so perform checks
        assert state_size > 0, "state_size must be > 0"  # behavior size
        self.state_size = state_size
        assert time > 0, "time must be > 0"  # length of an experiment
        self.time = time
        assert agent_n >= 2, "agent_n must be >= 2"  # number of agents
        self.agent_n = agent_n
        self.type = agent
        assert len(memory) == self.agent_n, 'memory must be a list, with as many entries as agent_n'
        for i in memory:
            assert type(i) == int, 'memory entries must be integers'
        self.memory = memory # memory of the agents
        assert len(behav_control) == self.agent_n, 'behav_control must be a list, with as many entries as agent_n'
        for i in behav_control:
            assert type(i) == int, 'behav_control entries must be integers'
        self.behav_control = behav_control # behavioral control of the agents

        ###### variables to be filled as the experiment runs
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
        n = self.agent_n
        while n:
            n -= 1
            if self.type[n-1] == "model":
                self.agents.append(Agent_with_Model(
                    self.state_size, float(self.memory[n-1]), float(self.behav_control[n-1])))
            elif self.type[n-1] == "model_sig":
                self.agents.append(Agent_with_Sigmoid_Model(
                    self.state_size, float(self.memory[n-1]), float(self.behav_control[n-1])))
            elif self.type[n-1] == "model_lin":
                self.agents.append(Agent_with_Linear_Model(
                    self.state_size, float(self.memory[n-1]), float(self.behav_control[n-1])))
            # elif self.type[n-1] == "chaos":
            #     self.agents.append(Agent_of_Chaos(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1])))
            # elif self.type[n-1] == "average":
            #     self.agents.append(Agent_Average(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1]), self.memory[n-1]))
            # elif self.type[n-1] == "prediction":
            #     self.agents.append(Agent_Average_Prediction(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1]), self.memory[n-1]))
            # elif self.type[n-1] == "dummy":
            #     self.agents.append(Agent_Dummy(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1])))
            # else:
            #     self.agents.append(Agent_Dummy(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1])))
            else:
                raise Exception("No valid agents could be found")

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
    parser.add_argument("-s", "--state_size", type=int,
                        metavar="state_size", help="size of behavior vector")
    parser.add_argument("-t", "--time", type=int,
                        metavar="time", help="number of time steps")
    parser.add_argument("-q", "--agent", type=str, nargs='+',
                        metavar="agent", help="type of agents to be used: chaos, average, dummy, model, prediction")
    parser.add_argument("-a", "--alpha", type=float, nargs='+',
                        metavar="alpha", help="prior learning rate: 0.001 - 1")
    parser.add_argument("-b", "--beta", type=float, nargs='+',
                        metavar="beta", help="conformity learning rate: 0.001 - 1")
    parser.add_argument("-r", "--seed", type=int, nargs='+',
                        metavar="seed", help="random seed for generating priors")
    parser.add_argument("-m", "--memory", type=int, nargs='+',
                        metavar="memory", help="how far back can an agent consider other's history")

    args = parser.parse_args()

    world = World(args.state_size, args.time,
                  args.agent, args.alpha, args.beta, args.seed, args.memory)
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
