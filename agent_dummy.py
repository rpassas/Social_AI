import numpy as np
import random


class Agent_Dummy():
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

    def __init__(self, state_size=3, alpha=0, beta=0, seed=666, inference_fn='IRL',  action_cost_fn='linear'):
        # size of a state
        if state_size < 0:
            self.state_size = 3
        else:
            self.state_size = state_size
        # behavioral priors
        self.b_priors = np.asarray([random.randint(0, 1)
                                   for i in range(self.state_size)])
        np.random.seed(seed)
        # current behavior
        self.behavior = []
        # estimate of world state parameters
        self.world_pred = np.random.rand(1, state_size).round(3)[0]
        # history of world states
        self.world = []
        # metabolic cost so far (accrued via learning)
        self.metabolism = 0.0
        # action cost function
        self.a_c_fn = action_cost_fn
        # function for estimating parameters

        # priors adjustment rate
        self.alpha = 0
        # estimates adjustment rate
        self.beta = 0

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        return self.b_priors

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
        dif = [np.asarray([abs(g-h)
                           for g, h in zip(self.world[-1], self.world_pred)])][0]
        e = round(np.sum(dif)/len(dif), 3)
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
            magnitude = random.choice([-1, 1]) * pred_error
            r = round(random.uniform(0, magnitude), 3) * self.alpha
            self.b_priors = [np.asarray([
                abs(i - r) if abs(i - r) <= 1 else i for i in self.b_priors])][0]
            self.metabolism += pred_error

    def learn_predict_world(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        # JT - is this adjusting to move self.world_pred closer to the previous self.world? This seems like it's updating randomly instead.
        '''
        Adjust prediction of world states based on prediction error.
        '''
        pred_error = self.behavior_prediction_error()
        threshold = self.action_cost(pred_error)
        if pred_error > threshold:
            b = random.choice([-1, 1]) * self.beta
            self.world_pred = [abs(i - b) if abs(i - b)
                               <= 1 else i for i in self.world_pred]

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
            return 0.2

        else:
            return 0.2

    def get_type(self):
        '''
        Get the agent type.
        '''
        return "dummy"

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
