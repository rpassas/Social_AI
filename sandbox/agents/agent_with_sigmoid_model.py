import numpy as np


class Agent_with_Sigmoid_Model():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention.

    INPUTS:
        state_size [integer, default=500]: sets size of behavior feature space, N. Expected behavior occurs when N is large.
        memory [integer >= 0, default=4]: sets memory weighting of prior predictions.
            Prediction error always adjusts from a prior prediction,
            so memory = 0 includes only weighting from the prior trial.
            memory > 0 weights the prior by earlier predictions, giving equal weight to each.
        behav_control [integer >= -1, default = 4]: sets weighting of prior behaviors.
            Received prediciton error will be linearly transformed into a new behavioral prior.
            If behav_control is set to -1, then the new behavioral prior is used without further weighting.
            If behav_control >=0 then earlier behavioral priors are averaged into the new one,
            behav_control = 0 includes the behavioral prior from trial t, and
            behav_control > 0 includes behavioral priors from trial t - behav_control.
        model_var [integer, default = 1]: sets the range of values present in the behavior model. When
            set to 0, the model is set to an identity matrix.



    VARIABLES:
        b_priors: N length vector, giving probability (to 3 decimal places) of
            agent DISPLAYING features of behavior on trial t.
        behavior: N length vector, indicating presence/absence (1/0) of agent's features of behavior trial t.
        past_priors: list of N length vectors, each recording b_prior for trial t.
        world_pred: N length vector, giving estimated probability (to 3 decimal places)
            of observing features of behavior FROM THE OTHER AGENT on trial t.
    """

    def __init__(self, state_size=500, memory=4, behav_control=4, model_var=1, action_cost_fn='linear'):
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
        assert memory >= 0, "memory must be >= 0"
        self.memory = int(memory)
        # how much of world is considered for current prediction
        assert behav_control >= 0, "behav_control must be >= -1"
        self.behav_control = behav_control
        # model_var or variance of the model determines the range of values in behav_model
        assert model_var >= 0, "model variance must be >= 0"
        self.model_var = model_var
        # behavioral model applies some randomness or "personality" to how behavior gets adjusted
        self.behav_model = np.random.rand(
            state_size, state_size)*self.model_var

        self.metabolism = 0.0  # metabolic cost so far (accrued via learning)
        self.a_c_fn = action_cost_fn  # action cost function

        self.attn = np.identity(self.state_size)  # attention matrix

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        self.past_priors.append(self.b_priors)
        return np.random.binomial(1, self.b_priors)

    def get_predictability(self):
        '''
        Returns the predictability of the agents where a score of 1 indicates priors 
        being close to 0 or 1 (taken as an average across predictability of each prior).
        '''
        predictability = [(abs(p - 0.5))*2 for p in self.b_priors]
        return sum(predictability)/len(predictability)

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
        dif = self.world_pred - \
            self.world[-1]  # array of differences, for each behavioral feature
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
        mem = int(min(self.behav_control, len(self.past_priors)))
        for m in range(2, mem+1):
            i = -1 * m
            sum_priors = [g + h for g,
                          h in zip(sum_priors, self.past_priors[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        top = sum_priors + matrix_sigmoid(self.behav_model @ attn_weighted_dif)
        self.b_priors = top / (mem + 2)

    def learn_predict_world(self):
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''
        sum_pred = self.past_predictions[-1]  # always include last memory from current trial t.
        mem = int(min(self.memory, len(self.past_predictions)))
        for m in range(2, mem+1):
            i = -1 * m
            sum_pred = [g + h for g,
                        h in zip(sum_pred, self.past_predictions[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        top = sum_pred - 2*(matrix_sigmoid(attn_weighted_dif)-0.5)
        self.world_pred = top / (mem + 1)

    def update_attention(self):
        attention = [-((p * math.log(p, 2)) + ((1-p) * math.log(1-p,  2)))
                     for p in self.world_pred]
        self.attn = np.diag(attention)
        '''
        mem = int(min(self.memory, len(self.world)))
        for m in range(1, mem+1):
            if m == 1:
                # grab first behavior to start the array.
                sum_world = self.world[-1]
                sum_pred = self.past_predictions[-1]
            else:
                i = -1 * m
                sum_world = [g + h for g,
                             h in zip(sum_world, self.world[i])]
                sum_pred = [g + h for g,
                            h in zip(sum_pred, self.past_predictions[i])]
        world_score = [2*((w / mem) - 0.5) ** 2 for w in sum_world]
        pred_score = [(p / mem) / 2 for p in sum_pred]
        for i in range(len(self.world[0])):
            self.attn[i][i] = world_score + pred_score
        '''

    def get_attention(self):
        return self.attn

    def get_costs(self):
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        return attn_weighted_dif

    def get_avg_costs(self):
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        avg_attn_dif = np.sum(abs(attn_weighted_dif))/len(attn_weighted_dif)
        return avg_attn_dif

    def get_total_cost(self):
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
        return "model_sig"


def matrix_sigmoid(x):
    '''
    Helper sigmoid function
    '''
    # print(x)
    return 1 / (1 + np.exp(-x))
