import numpy as np
import random


class Basic_Agent():
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

    def __init__(self, state_size=5, alpha=1, beta=0.5, inference_fn='IRL',  action_cost_fn='linear'):
        # size of a state
        self.state_size = state_size
        # behavioral priors
        self.b_priors = np.random.rand(1, state_size).round(3)
        # current behavior
        self.behavior = []
        # estimate of world state parameters
        self.world_pred = self.b_priors
        ##JT - predictions about the other agent are set to be equal to the agent's behavioral priors. We should set these independently I think.
        ## So that would just be np.random.rand(1, state_size).round(3) again.
        # history of world states
        self.world = []
        # metabolic cost so far (accrued via learning)
        self.metabolism = 0.0
        # action cost function
        self.a_c_fn = action_cost_fn
        # function for estimating parameters

        # priors adjustment rate
        ##JT - Should alpha and beta be set to 0.001 if < .001 ?
        if alpha > 1 or alpha < 0.001:
            self.alpha = 1
        else:
            self.alpha = alpha
        # estimates adjustment rate
        if beta > 1 or beta < 0.001:
            self.beta = 1
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
        print(self.world[-1])
        print(self.world_pred)
        dif = [np.asarray([abs(g-h)
                           for g, h in zip(self.world[-1][0], self.world_pred[0])])]
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
            a = random.choice([-1, 1]) * pred_error * self.alpha
            r = round(random.uniform(a, 1), 3)
            self.b_priors = [np.asarray([
                abs(i - r) if abs(i - r) <= 1 else i for i in self.b_priors[0]])]
            self.metabolism += pred_error

    def learn_predict_world(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        ##JT - is this adjusting to move self.world_pred closer to the previous self.world? This seems like it's updating randomly instead.
        '''
        Adjust prediction of world states based on prediction error.
        '''
        pred_error = self.behavior_prediction_error()
        threshold = self.action_cost(pred_error)
        if pred_error > threshold:
            b = random.choice([-1, 1]) * self.beta
            self.world_pred = [[abs(i - b) if abs(i - b)
                               <= 1 else i for i in self.world_pred[0]]]
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
        ##JT - I like the idea that we'll need a threshold, and I think one objective simple objective might be for agents to 
        ## learn when to stop updating self.world_pred to avoid overfitting, since priors (which range from 0-1) will never perfectly match behavior 
        ## (which is operationalized as 0 or 1). We might want to set this low at first though, to see agents bounce around a bit.
        if self.a_c_fn == "linear":
            return 0.5

        else:
            return 0.5
