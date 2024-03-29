from .agents.agent_of_chaos import Agent_of_Chaos
#from old_versions.agent_average_behavior import Agent_Average
from .agents.agent_average_prediction import Agent_Average_Prediction
from .agents.agent_dummy import Agent_Dummy
#from old_versions.agent_with_model import Agent_with_Model
from .agents.agent_with_sigmoid_model import Agent_with_Sigmoid_Model
#from old_versions.agent_with_linear_model import Agent_with_Linear_Model
from .agents.agent_with_alt_sigmoid_model import Agent_with_Alt_Sigmoid_Model
from .agents.agent_bayes import Agent_Bayes
from .agents.agent_semi_bayes import Agent_Semi_Bayes
from .agents.agent import Agent
import numpy as np
import argparse


class World():
    """
    World holds state data.
    INPUTS:
        state_size [integer, default=3]: sets size of behavior feature space, N.
        time [integer, default=100]: sets number of experimental trials, t.
        agent [strings, default = "model_alt"]: selects agent to use. Each has different properties. See agents documentation.
        seed [integer, default=None]: use an integer seed in order to replicate analyses.
        memory [integer >= 0, default=0]: new prediction is an average of N prior predictions +
            the updated prediction from the current trial..
        behav_control [integer >= 0, default = 0]: new behavior is an average of N prior behaviors +
            the new behavior resulting from prediction error on the current trial..
        model_var [integer, default = 1]: sets the range of values present in the behavior model. When
            set to 0, the model is a matrix of zeroes, meaning behavior does not change from its initial setting.
        behav_initial_spread [integer, default = 1]: multiplier applied within sigmoid to initial behavioral_priors.
            High values create a bimodial distribution. Zero gives 0.5 for all initial behavioral_priors.
        pred_initial_spread [integer, default = 1]: multiplier applied within sigmoid to initial predictions.
                High values create a bimodial distribution. Zero gives 0.5 for all initial predictions.
        change_points = [lists of integers, default = None]: reinitialize behavior at time = change_point.

        agent_n [integer, default=2]: sets number of agents. Currently only set-up to handle 2.
    """

    def __init__(self, state_size=3, time=100, agent=["base", "base"],
                 seed=None, model_var=[[0, 1], [0, 1]], behav_initial_spread=[1, 1], init_priors=[], init_preds=[],
                 pred_initial_spread=[1, 1], pred_a=[0, 0], behav_a=[0, 0], change_points=[[None], [None]], agent_n=2,
                 prediction=["sigmoid", "sigmoid"], behavior=["sigmoid", "sigmoid"], attention=["static", "static"]):
        if seed:
            np.random.seed(seed)
        # argparse will make unfilled optional args 'None', so perform checks
        assert state_size > 0, "state_size must be > 0"  # behavior size
        self.state_size = state_size
        if init_priors:
            for p in init_priors:
                assert len(
                    p) == self.state_size, "state size and prior size must match"
            self.init_priors = init_priors
        else:
            self.init_priors = [[], []]
        if init_preds:
            for p in init_preds:
                assert len(
                    p) == self.state_size, "state size and prediction size must match"
            self.init_preds = init_preds
        else:
            self.init_preds = [[], []]
        assert time > 0, "time must be > 0"  # length of an experiment
        self.time = time
        assert agent_n >= 2, "agent_n must be >= 2"  # number of agents
        self.agent_n = agent_n
        self.type = agent
        # assert len(
        #    memory) == self.agent_n, 'memory must be a list, with as many entries as agent_n'
        # for i in memory:
        #    assert type(i) == int, 'memory entries must be integers'
        self.memory = [0, 0]  # memory of the agents
        # assert len(
        #    behav_control) == self.agent_n, 'behav_control must be a list, with as many entries as agent_n'
        # for i in behav_control:
        #    assert type(i) == int, 'behav_control entries must be integers'
        self.pred_a = []
        for a in pred_a:
            assert a >= 0, "model variance must be at least 0"
            assert a <= 1, "model variance must be at most 1"
            self.pred_a.append(a)
        self.behav_a = []
        for a in behav_a:
            assert a >= 0, "model variance must be at least 0"
            assert a <= 1, "model variance must be at most 1"
            self.behav_a.append(a)
        self.behav_control = [0, 0]  # behavioral control of the agents
        # for bound in model_var:
        #    for i in bound:
        #        assert i >= 0, "model variance must be >= 0"
        for i in change_points:
            assert isinstance(
                i, list), 'each entry in change_points must be a list of intergers'
        self.change_points = change_points
        self.model_var = model_var  # behavioral control of the agents
        # adjust bimodal distribution of b_priors
        self.behav_initial_spread = behav_initial_spread
        # adjust bimodal distribution of predictions
        self.pred_initial_spread = pred_initial_spread
        # prediction, behavior, and attention update mechanisms
        self.prediction = prediction
        self.behavior = behavior
        self.attention = attention
        # variables to be filled as the experiment runs
        self.agents = []
        '''
        self.b_priors = np.empty((self.agent_n, self.state_size))
        self.behaviors = np.empty((self.agent_n, self.state_size))
        self.predictions = np.empty((self.agent_n, self.state_size))
        self.errors = np.empty((self.agent_n, self.state_size))
        self.avg_predictability = np.empty((self.agent_n, 1))
        '''
        self.b_priors = [[] for a in range(self.agent_n)]
        self.behaviors = [[] for a in range(self.agent_n)]
        self.predictions = [[] for a in range(self.agent_n)]
        self.errors = [[] for a in range(self.agent_n)]
        self.avg_predictability = [[] for a in range(self.agent_n)]
        self.concur = [[] for a in range(self.agent_n)]
        self.costs = [[] for a in range(self.agent_n)]
        self.avg_abs_error = [[] for a in range(self.agent_n)]
        self.tru_dif = [[] for a in range(self.agent_n)]

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
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1]))
            elif self.type[n-1] == "base":
                self.agents.append(Agent(state_size=self.state_size, model_var=self.model_var[n-1],
                                         init_preds=self.init_preds[n -
                                                                    1], init_priors=self.init_priors[n-1],
                                         pred_a=self.pred_a[n - 1], behav_a=self.behav_a[n -
                                                                                         1], prediction=self.prediction[n-1],
                                         behavior=self.behavior[n-1], attention=self.attention[n-1]))
            elif self.type[n-1] == "bayes":
                self.agents.append(Agent_Bayes(
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1]))
            elif self.type[n-1] == "semi_bayes":
                self.agents.append(Agent_Semi_Bayes(
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1]))
            elif self.type[n-1] == "model_sig":
                self.agents.append(Agent_with_Sigmoid_Model(
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1]))
            elif self.type[n-1] == "model_lin":
                self.agents.append(Agent_with_Linear_Model(
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1]))
            elif self.type[n-1] == "model_alt":
                self.agents.append(Agent_with_Alt_Sigmoid_Model(
                    state_size=self.state_size, memory=float(self.memory[n-1]),
                    behav_control=float(self.behav_control[n-1]), model_var=self.model_var[n-1],
                    behav_initial_spread=float(self.behav_initial_spread[n-1]),
                    pred_initial_spread=float(self.pred_initial_spread[n-1])))
            elif self.type[n-1] == "chaos":
                self.agents.append(Agent_of_Chaos(
                    state_size=self.state_size, alpha=0.5, beta=0.5))
            # elif self.type[n-1] == "average":
            #     self.agents.append(Agent_Average(
            #         self.state_size, float(self.alphas[n-1]), float(self.betas[n-1]), self.memory[n-1]))
            elif self.type[n-1] == "prediction":
                self.agents.append(Agent_Average_Prediction(
                    state_size=self.state_size, alpha=(self.model_var[n-1]*0.1), beta=0.5, memory=self.memory[n-1]))
            elif self.type[n-1] == "dummy":
                self.agents.append(Agent_Dummy(
                    state_size=self.state_size, alpha=0.5, beta=0.5))
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
        '''
        # run first iteration
        for i in range(len(self.agents)):
            b = self.agents[i].make_behavior()
            p = self.agents[i].get_behav_priors()
            predictability = [abs(x-0.5)*2 for x in p]
            avg_pred = sum(predictability)/len(predictability)
            self.avg_predictability[i] = avg_pred
            self.behaviors[i] = b
            self.b_priors[i] = p
        print("b:", self.behaviors)
        for i in range(len(self.agents)):
            if i == 0:
                # agent 0 gets agent 1's behavior
                self.agents[i].get_world(self.behaviors[1][-1])
            else:
                # agent 1 gets agent 0's behavior
                self.agents[i].get_world(self.behaviors[0][-1])
            p = self.agents[i].make_prediction()
            dif, avg_abs_error = self.agents[i].behavior_prediction_error()
            self.agents[i].learn_conform()
            self.agents[i].learn_predict_world()
            self.predictions[i] = p
            self.errors[i] = dif
            self.avg_abs_error[i] = avg_abs_error
        '''
        # rest of the trials
        while time_left:
            # generate behaviors
            current_time = self.time - time_left
            for i in range(len(self.agents)):
                if current_time in self.change_points[i]:
                    self.agents[i].new_behavior()
                b = self.agents[i].make_behavior()
                p = self.agents[i].get_behav_priors()
                a_p = self.agents[i].get_predictability()
                #predictability = [abs(x-0.5)*2 for x in p]
                #avg_pred = sum(predictability)/len(predictability)
                # self.avg_predictability[i].append(avg_pred)
                self.behaviors[i].append(b)
                self.b_priors[i].append(p)
                self.avg_predictability[i].append(a_p)

            # receive behaviors, predict, learn
            # will have to be updated for multi agents
            for i in range(len(self.agents)):
                if i == 0:
                    # agent 0 gets agent 1's behavior
                    self.agents[i].get_world(self.behaviors[1][-1])
                else:
                    # agent 1 gets agent 0's behavior
                    self.agents[i].get_world(self.behaviors[0][-1])
                p = self.agents[i].make_prediction()
                dif, avg_abs_error = self.agents[i].behavior_prediction_error(
                    1)
                c = self.agents[i].get_avg_costs()
                self.agents[i].learn_conform()
                self.agents[i].learn_predict_world()
                self.agents[i].update_attention()
                # record results
                self.costs[i].append(c)
                self.predictions[i].append(p)
                self.errors[i].append(dif)
                self.avg_abs_error[i].append(avg_abs_error)
                #print("COST:", self.avg_abs_error)
                # print("\n")
            for i in range(len(self.agents)-1):
                for j in range(i+1, len(self.agents)):
                    self.concur[i].append(1 - (
                        abs(abs(self.avg_abs_error[i][-1]) - abs(self.avg_abs_error[j][-1]))))
                    self.concur[j].append(1 - (
                        abs(abs(self.avg_abs_error[i][-1]) - abs(self.avg_abs_error[j][-1]))))
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
        #error_array = np.array(self.errors)
        #error_T = error_array.T
        return self.errors

    def get_avg_abs_error(self):
        '''
        Get a representation of the avg_abs_error so each list is an agents cost across time (cumulative error).
        '''
        #cost_array = np.array(self.avg_abs_error)
        #cost_T = cost_array.T
        return self.avg_abs_error

    def get_pred(self):
        '''
        Get a representation of the predictions so each list is an agents prediction across time.
        '''
        #pred_array = np.array(self.predictions)
        #pred_T = pred_array.T
        return self.predictions

    def get_behav_priors(self):
        '''
        Get a representation of the priors so each list is an agent's behavioral priors across time.
        '''
        #prior_array = np.array(self.b_priors)
        #prior_T = prior_array.T
        return self.b_priors

    def get_behaviors(self):
        '''
        Get all the behaviors of the experiment.
        '''
        return self.behaviors

    def get_predictability(self):
        '''
        Get the average predictability score of the priors of each agent:
        1 = predictable (0 or 1 prior)
        ...
        0 = unpredictable (0.5 prior)
        '''
        return self.avg_predictability

    def get_concurrence(self):
        '''
        Get the 1 - average absolute difference between errors i.e. "concurrence"
        1 = errors match
        ...
        0 = errors are a complete mismatch
        '''
        return self.concur

    def get_costs(self):
        '''
        Get the average predictability score of the priors of each agent:
        1 = predictable (0 or 1)
        ...
        0 = unpredictable (0.5)
        '''
        return self.costs

    def get_tru_dif(self):
        '''
        '''
        for t in range(self.time):
            self.tru_dif[0].append(
                sum(abs(self.b_priors[1][t]-self.predictions[0][t]))/self.state_size)
            self.tru_dif[1].append(
                sum(abs(self.b_priors[0][t]-self.predictions[1][t]))/self.state_size)
        return self.tru_dif

    def print_results(self):
        '''
        Print the results of the experiment.
        '''
        for a in self.agents:
            print("agent type:   {}".format(a.get_type()))
            #print("agent alpha:   {}".format(a.get_alpha()))
            #print("agent beta:   {}".format(a.get_beta()))
            print("  ---  ")

        print("\n")
        for t in range(self.time):
            print("time step:   {}".format(t+1))
            for a in range(self.agent_n):
                print("bhv priors:  {}".format(
                    [b.tolist() for b in np.round(self.b_priors[a][t], 3)]))
                print("behaviors:   {}".format(
                    [b.tolist() for b in self.behaviors[a][t]]))
                print("predictions: {}".format(
                    [b.tolist() for b in np.round(self.predictions[a][t], 3)]))
                print("errors:      {}".format(self.errors[a][t]))
                print("net avg_abs_error:   {}".format(
                    np.round(self.avg_abs_error[a][t], 3)))
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
