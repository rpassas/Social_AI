import numpy as np


class Agent_with_Model():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention.

    INPUTS:
        state_size [integer, default=3]: sets size of behavior feature space, N.


    VARIABLES:
        b_priors: N length vector, giving probability (to 3 decimal places) of
            agent DISPLAYING features of behavior on trial t.
        behavior: N length vector, indicating presence/absence (1/0) of agent's features of behavior trial t.
        past_priors: list of N length vectors, each recording b_prior for trial t.
        world_pred: N length vector, giving estimated probability (to 3 decimal places)
            of observing features of behavior FROM THE OTHER AGENT on trial t.
    """

    def __init__(self, state_size=3, alpha=1, beta=1, seed=None, memory=4, inference_fn='IRL',  action_cost_fn='linear'):
        assert state_size > 0, "state_size must be > 0"
        self.state_size = state_size # size of a state
        self.b_priors = np.random.rand(1, state_size).round(3)[0] # generates a new instance of a behavioral prior.
        self.past_priors = []  # stores past behavioral priors.
        self.behavior = []  # current behavior. I THINK THIS GOES UNUSED?
        self.world_pred = np.random.rand(1, state_size).round(3)[0]  # estimate of world state parameters
        self.past_predictions = []  # past predictions
        self.world = []  # history of world states
        if memory < 0:  # how much of world is considered for current prediction
            self.memory = 1
        elif memory > 50:
            self.memory = 50
        else:
            self.memory = memory
        self.metabolism = 0.0  # metabolic cost so far (accrued via learning)
        self.a_c_fn = action_cost_fn  # action cost function
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
        # attention matrix
        self.attn = np.identity(self.state_size)

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        self.past_priors.append(self.b_priors)
        return np.random.binomial(1, self.b_priors)

    def make_prediction(self):
        '''
        Generate actual world prediction (list of 0/1) from priors
        '''
        return self.world_pred

    def get_world(self, world):
        '''
        For adding current world state to the history
        '''
        self.world.append(world)

    def get_behav_priors(self):
        '''
        Gets the behavioral priors of the agent.
        '''
        return self.b_priors

    def behavior_prediction_error(self):
        '''
        Given the current state of the world, how off was the agent's prediction? (i.e. how well do we predict the world?)
        Currently just the difference
        '''
        dif = [np.asarray([g-h
                           for g, h in zip(self.world_pred, self.world[-1])])][0]
        print(dif)
        e = round(np.sum(dif)/len(dif), 3)
        return dif

    def behav_update(self, pred_err, attn_matrix, behav_control, internal_model):
        '''
        Adjust behavioral priors to match the world state based on conformity error
        pred_err = prediction error at time t
        attn_matrix = attention matrix at time t
        behav_control = SET PARAMETER, weighting M previous trials into output behavior.
        internal model = linear transfer function from input information to output behavioral prior.
        '''
        sum_priors = self.past_priors[-1]
        # handle case where not yet enough trials.
        if len(self.past_priors) < behav_control:
            behav_control = len(self.past_priors)
        # START HERE - we're solving hte problem of how to take the average over vectors in np.
        for m in range(2, mem):
            i = -1 * m
            sum_priors = [g + h for g,
                          h in zip(sum_priors, self.past_priors[i])]
        e = self.behavior_prediction_error()
        print(e)
        print(self.attn)
        exp = self.attn @ e
        print(exp)
        top = sum_priors + matrix_sigmoid(exp)
        self.b_priors = top / (mem + 1)

    def learn_conform(self):
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''
        sum_priors = self.past_priors[-1]
        stability = int(self.alpha * 10)
        mem = min(stability, len(self.past_priors))
        for m in range(2, mem):
            i = -1 * m
            sum_priors = [g + h for g,
                          h in zip(sum_priors, self.past_priors[i])]
        e = self.behavior_prediction_error()
        print(e)
        print(self.attn)
        exp = self.attn @ e
        print(exp)
        top = sum_priors + matrix_sigmoid(exp)
        self.b_priors = top / (mem + 1)

    def learn_predict_world(self):
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''
        sum_pred = self.past_predictions[-1]
        mem = min(self.memory, len(self.past_predictions))
        for m in range(2, mem):
            i = -1 * m
            sum_pred = [g + h for g,
                        h in zip(sum_pred, self.past_predictions[i])]
        e = self.behavior_prediction_error()
        exp = self.attn @ e
        top = sum_pred + matrix_sigmoid(exp)
        self.world_pred = top / (mem + 1)

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
            return 0.4

        else:
            return 0.4

    def get_type(self):
        '''
        Get the agent type.
        '''
        return "model"

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


def matrix_sigmoid(x):
    '''
    Helper sigmoid function
    '''
    print(x)
    return 1 / (1 + np.exp(-x))
