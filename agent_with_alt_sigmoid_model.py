import numpy as np


class Agent_with_Alt_Sigmoid_Model():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention to particular 
    features within a given state.

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

    def __init__(self, state_size=3, alpha=1, beta=1, seed=None, memory=4, behav_control=4, inference_fn='IRL',  action_cost_fn='linear'):
        assert state_size > 0, "state_size must be > 0"
        self.state_size = state_size  # size of a state
        # generates a new instance of a behavioral prior.
        self.b_priors = np.random.rand(1, state_size).round(3)[0]
        self.past_priors = []  # stores past behavioral priors.
        self.behavior = []  # current behavior. I THINK THIS GOES UNUSED?
        self.world_pred = np.random.rand(1, state_size).round(
            3)[0]  # estimate of world state parameters
        self.past_predictions = []  # past predictions
        self.world = []  # history of world states
        # how much of world is considered for current prediction
        assert memory > 0, "memory must be > 0"
        self.memory = memory
        self.behav_control = behav_control

        # This is necessary to get people to change their behaviors, or else they'll just remain the same.
        # It doesn't seem to move behavior very much right now though, so we may need to experiment with values.
        # self.behav_model = np.random.randint(-1,
        #                                     1, size=(state_size, state_size))
        self.behav_model = np.identity(self.state_size)

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
        self.attn = np.identity(self.state_size)  # attention matrix

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        self.past_priors.append(self.b_priors)
        return np.random.binomial(1, self.b_priors)

    def make_prediction(self):
        '''
        Generate actual world prediction
        '''
        self.past_predictions.append(self.world_pred)
        return self.world_pred

    def get_world(self, world):
        '''
        For adding current world state (i.e. other agent's behavior) to the history
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
        Returns vector of +/- prediciton error, and average absolute prediction error
        '''
        dif = self.world[-1] - \
            self.world_pred  # array of differences, for each behavioral feature
        # absolute prediction error across all features
        avg_abs_error = round(np.sum(abs(dif))/len(dif), 3)
        return dif, avg_abs_error

    # def behav_update(self, pred_err, attn_matrix, behav_control, internal_model):
    #     '''
    #     Adjust behavioral priors to match the world state based on conformity error
    #     pred_err = prediction error at time t
    #     attn_matrix = attention matrix at time t
    #     behav_control = SET PARAMETER, weighting M previous trials into output behavior.
    #     internal model = linear transfer function from input information to output behavioral prior.
    #     '''
    #     sum_priors = self.past_priors[-1]
    #     # handle case where not yet enough trials.
    #     if len(self.past_priors) < behav_control:
    #         behav_control = len(self.past_priors)
    #     # START HERE - we're solving hte problem of how to take the average over vectors in np.
    #     for m in range(2, mem):
    #         i = -1 * m
    #         sum_priors = [g + h for g,
    #                       h in zip(sum_priors, self.past_priors[i])]
    #     dif, avg_abs_error = self.behavior_prediction_error()
    #     # print(dif)
    #     # print(self.attn)
    #     attn_weighted_dif = self.attn @ dif
    #     # print(attn_weighted_dif)
    #     top = sum_priors + matrix_sigmoid(self.behav_model @ attn_weighted_dif) # multiply by internal model to get new behavior.
    #     self.b_priors = top / (mem + 1)

    def learn_conform(self):
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''
        if self.behav_control < 0:
            sum_priors = 0
        else:
            # get first vector, if self control used.
            sum_priors = self.past_priors[-1]
        mem = min(self.behav_control, len(self.past_priors))
        for m in range(2, mem+1):
            i = -1 * m
            sum_priors = [g + h for g,
                          h in zip(sum_priors, self.past_priors[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        top = sum_priors + matrix_sigmoid(self.behav_model @ attn_weighted_dif)
        self.b_priors = top / (mem + 1)

    def learn_predict_world(self):
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''
        sum_pred = self.past_predictions[-1]  # always include last memory from current trial t.
        mem = min(self.memory, len(self.past_predictions))
        for m in range(2, mem+1):
            i = -1 * m
            sum_pred = [g + h for g,
                        h in zip(sum_pred, self.past_predictions[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        #base = np.asarray(sum_pred) / mem
        #sig = dynamic_sigmoid(base, attn_weighted_dif)
        sig = dynamic_sigmoid(self.past_predictions[-1], attn_weighted_dif)
        print("world, pred:")
        print(self.world[-1])
        print(self.past_predictions[-1])
        print("error:")
        print(dif)
        print("sig:")
        print(sig)
        #sig = matrix_sigmoid(attn_weighted_dif)
        top = sum_pred + sig
        p = top / (mem + 1)
        print("update:")
        print(self.past_predictions[-1])
        print(p)
        print("---")
        self.world_pred = p

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
    Helper sigmoid function where the intercept is 0.5
    '''
    return 1 / (1 + np.exp(-x))


def dynamic_sigmoid(i, x):
    '''
    Helper sigmoid function where the intercept is a value of i (list)
    '''
    y = np.exp(-x)
    out = np.asarray([1 / (1 + ((1/i[j]) - 1) * y[j]) for j in range(len(i))])
    return out


def stabilize(x):
    '''
    Probably not useful - but keeping it here for now in case it becomes so.
    Yields magnitude value for x between 0 and 1 where the intercept is 0.
    '''
    y = np.array([(-0.5**i)+1 if i > 0 else (-2**i) + 1 for i in x])
    return y
