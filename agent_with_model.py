import numpy as np
import random


class Agent_with_Model():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention.

    """
    # TODO: implement inference and cost functions in separate classes
    # inference or parameter estimation should be biased towards most recent states and take error for each state into account

    def __init__(self, state_size=3, alpha=1, beta=1, seed=666, memory=4, inference_fn='IRL',  action_cost_fn='linear'):
        # size of a state
        if state_size < 0:
            self.state_size = 3
        else:
            self.state_size = state_size
        np.random.seed(seed)
        # behavioral priors
        self.b_priors = np.random.rand(1, state_size).round(3)[0]
        # current behavior
        self.behavior = []
        # estimate of world state parameters
        self.world_pred = np.random.rand(1, state_size).round(3)[0]
        # past predictions
        self.past_predictions = []
        # history of world states
        self.world = []
        # how much of world is considered for current prediction
        if memory < 0:
            self.memory = 1
        elif memory > 50:
            self.memory = 50
        else:
            self.memory = memory
        # metabolic cost so far (accrued via learning)
        self.metabolism = 0.0
        # action cost function
        self.a_c_fn = action_cost_fn
        # function for estimating parameters

        # priors adjustment rate
        if alpha > 1 or alpha < 0:
            self.alpha = 1
        elif alpha < 0:
            self.alpha = 0.01
        else:
            self.alpha = alpha
        # estimates adjustment rate
        if beta > 1 or beta < 0:
            self.beta = 1
        elif beta < 0:
            self.beta = 0.01
        else:
            self.beta = beta

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        return np.random.binomial(1, self.b_priors)

    def make_prediction(self):
        '''
        Generate actual world prediction (list of 0/1) from priors
        '''
        p = np.random.binomial(1, self.world_pred)
        self.past_predictions.append(p)
        return p

    def get_world(self, world):
        '''
        For adding current world state to the history
        '''
        self.world.append(world)

    def get_priors(self):
        '''
        Gets the behavioral priors of the agent.
        '''
        return self.b_priors

    def behavior_prediction_error(self):
        '''
        Given the current state of the world, how off was the agent's prediction? (i.e. how well do we predict the world?)
        '''

    def learn_conform(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''

    def learn_predict_world(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''

    def get_cost(self):
        '''
        Get the cost so far.
        '''
        return self.metabolism

    def action_cost(self, error):
        '''
        Determines willingness to learn based on the cost (error)
        '''
        # JT - I like the idea that we'll need a threshold, and I think one objective simple objective might be for agents to
        # learn when to stop updating self.world_pred to avoid overfitting, since priors (which range from 0-1) will never perfectly match behavior
        # (which is operationalized as 0 or 1). We might want to set this low at first though, to see agents bounce around a bit.
        if self.a_c_fn == "linear":
            return 0.4

        else:
            return 0.4

    def get_type(self):
        '''
        Get the agent type.
        '''
        return "average"

    def get_alpha(self):
        '''
        Get the alpha value.
        '''
        return self.alpha

    def get_beta(self):
        '''
        Get the beta value.
        '''
        return self.beta


'''
    def directional_error(self):
        #dif = [abs(g-h) for g, h in zip(self.world[-1], self.make_prediction())]
        dif = [g-h for g, h in zip(self.world[-1], self.world_pred)]
        e = round(np.sum(dif)/len(dif[0]), 3)
        return e
'''
