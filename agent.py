import numpy as np
import random


class Agent():
    """
    This general agent has an action-cost curve determining likelihood of action given its cost,
    size of state (determines behavior richness), and an inference model for guessing the behavioral priors of other agents whose
    behavior states are observable.

    What: keep metabolic costs (learning/action cost) low while still learning to accurately predict external states

    How: try to predict their prediction of your behavior and align them so that their behavior is less likely to change
    and is easier to predict (i.e. learn their expectations of you and learn their behavior priors)

    """
    # TODO: implement inference and cost functions in separate classes
    # inference or parameter estimation should be biased towards most recent states and take error for each state into account

    def __init__(self, state_size=3, inference_fn='IRL',  action_cost_fn='linear'):
        # size of a state
        self.state_size = state_size
        # behavioral priors
        self.b_priors = np.random.rand(1, state_size).round(3)
        # current behavior
        self.behavior = []
        # estimate of world state parameters
        self.world_pred = self.b_priors.round(3)
        # history of world states
        self.world = []
        # metabolic cost so far (accrued via learning)
        self.metabolism = 0.0
        # action cost function
        self.a_c_fn = action_cost_fn
        # function for estimating parameters

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        return np.random.binomial(1, self.b_priors)

    def make_prediction(self):
        '''
        Generate actual world prediction (list of 0/1) from priors
        '''

        # return np.random.binomial(1, self.world_pred)
        return self.world_pred

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
        #dif = [abs(g-h) for g, h in zip(self.world[-1], self.make_prediction())]
        dif = [abs(g-h) for g, h in zip(self.world[-1], self.world_pred)]
        e = round(np.sum(dif)/len(dif[0]), 3)
        return e

    def learn_conform(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''
        pred_error = self.behavior_prediction_error()
        threshold = self.action_cost(pred_error)
        if pred_error > threshold:
            r = round(random.uniform(0.5, pred_error), 3)
            self.b_prior = [abs(r - i) for i in self.world_pred]
            self.metabolism += pred_error

    def learn_predict_world(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust prediction of world states based on prediction error.
        '''
        pred_error = self.behavior_prediction_error()
        threshold = self.action_cost(pred_error)
        if pred_error > threshold:
            r = round(random.uniform(0.7, pred_error), 3)
            self.world_pred = [abs(r - i) for i in self.world_pred]
            self.metabolism += pred_error

    def get_cost(self):
        '''
        Get the cost so far.
        '''
        return self.metabolism

    def action_cost(self, error):
        '''
        Determines willingness to learn based on the cost (error)
        '''
        if self.a_c_fn == "linear":
            return 0.5
        else:
            return 0.5
