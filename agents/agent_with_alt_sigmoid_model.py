import numpy as np


class Agent_with_Alt_Sigmoid_Model():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention to particular
    features within a given state.

    INPUTS:
        state_size [integer, default=3]: sets size of behavior feature space, N.
        memory [integer >= 0, default=4]: sets memory weighting of prior predictions.
            Memory = 0 uses no weighting.
            memory > 0 weights the prior by earlier predictions, giving equal weight to each.
                At memory == 1, the current trial t is averaged with the new prediction.
        behav_control [integer >= 0, default = 4]: sets weighting of prior behaviors.
            Received prediciton error will be linearly transformed into a new behavioral prior.
            If behav_control is set to 0, then the new behavioral prior is used without further weighting.
            behav_control = 1 includes the behavioral prior from trial t.
            behav_control > 1 includes additional behavioral priors.
        model_var [integer, default = 0]: sets the range of values present in the behavior model. When
            set to 0, the model is a matrix of zeroes, meaning behavior does not change from its initial setting.
        learnable [integer, default = 1]: multiplier applied within sigmoid to initial behavioral_priors.
            High values create a bimodial distribution. Zero gives 0.5 for all initial behavioral_priors.


    VARIABLES:
        b_priors: N length vector, giving probability (to 3 decimal places) of
            agent DISPLAYING features of behavior on trial t.
        behavior: N length vector, indicating presence/absence (1/0) of agent's features of behavior trial t.
        past_priors: list of N length vectors, each recording b_prior for trial t.
        world_pred: N length vector, giving estimated probability (to 3 decimal places)
            of observing features of behavior FROM THE OTHER AGENT on trial t.
    """

    def __init__(self, state_size=3, seed=None, memory=0, behav_control=0, model_var=1, learnable=1, inference_fn='IRL',  action_cost_fn='linear'):
        assert state_size > 0, "state_size must be > 0"
        self.state_size = state_size  # size of a state
        # generates a new instance of a behavioral prior.
        self.b_priors = np.random.rand(1, self.state_size).round(3)[0]
        self.b_learnable = learnable # self.b_learnable (>=0) adjusts bimodal distribution of initial behavioral priors. # Allow this to be specified in World.
        assert learnable >= 0, "learnable must be >= 0"
        # values near 0 set most behaviors near 0.5. High values (e.g. 10) set clear bimodal distribution. Genrally use values in [0, 10] range.
        self.b_priors = matrix_sigmoid((2*self.b_priors-1)*self.b_learnable)
        self.past_priors = []  # stores past behavioral priors.
        self.behavior = []  # current behavior. I THINK THIS GOES UNUSED?
        self.world_pred = np.random.rand(1, self.state_size).round(
            3)[0]  # estimate of world state parameters
        self.past_predictions = []  # past predictions
        self.world = []  # history of world states
        # how much of world is considered for current prediction
        assert memory >= 0, "memory must be >= 0"
        self.memory = memory
        self.behav_control = behav_control
        # model_var or variance of the model determines the range of values in behav_model
        assert model_var >= 0, "model variance must be at least 0"
        self.model_var = model_var
        # behavioral model applies some randomness or "personality" to how behavior gets adjusted
        self.behav_model = (2*np.random.rand(self.state_size, self.state_size)-1)*self.model_var
        # model_thresh creates distributions where some input changes behavior drastically, while others have small effects.
        # TODO - parameterize this. It's a neat idea, but I don't know how it works yet.
        # self.model_thresh = .95
        # self.behav_model[abs(self.behav_model) > self.model_thresh] = self.behav_model[abs(self.behav_model) > self.model_thresh]*10
        # self.behav_model[abs(self.behav_model) <= self.model_thresh] = self.behav_model[abs(self.behav_model) <= self.model_thresh]*.1

        self.metabolism = 0.0  # metabolic cost so far (accrued via learning)
        self.a_c_fn = action_cost_fn  # action cost function
        self.attn = np.identity(self.state_size)  # attention matrix

    def new_behavior(self):
        '''
        Create new random behavior, initialized as the first behavioral prior was.
        '''
        self.b_priors = np.random.rand(1, self.state_size).round(3)[0]
        self.b_priors = matrix_sigmoid((2*self.b_priors-1)*self.b_learnable)

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        self.past_priors.append(self.b_priors)
        # print(np.round(self.b_priors, 5))
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
        if len(self.world[-1]) != len(self.world_pred):
            raise ValueError("state sizes between agents must match")
        dif = self.world[-1] - \
            self.world_pred  # array of differences, for each behavioral feature
        avg_abs_error = round(np.sum(abs(dif))/len(dif), 3)
        return dif, avg_abs_error

    def learn_conform(self):
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''
        mem = int(min(self.behav_control, len(self.past_priors)))
        if mem == 0:
            sum_priors = 0
        else:
            for m in range(1, mem+1):
                if m == 1:
                    sum_priors = self.past_priors[-1]
                else:
                    i = -1 * m
                    sum_priors = [g + h for g,
                                  h in zip(sum_priors, self.past_priors[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        updated_dif = self.behav_model @ attn_weighted_dif
        top = sum_priors + dynamic_sigmoid(self.past_priors[-1], updated_dif)
        self.b_priors = top / (mem + 1)

    def learn_predict_world(self):
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''
        mem = int(min(self.memory, len(self.past_predictions)))
        if mem == 0:
            sum_pred = 0
        else:
            for m in range(1, mem+1):
                if m == 1:
                    sum_pred = self.past_predictions[-1] # grab first prior prediction to start the array.
                else:
                    i = -1 * m
                    sum_pred = [g + h for g,
                                h in zip(sum_pred, self.past_predictions[i])]
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        top = sum_pred + dynamic_sigmoid(self.past_predictions[-1], attn_weighted_dif)
        # print(top)
        self.world_pred = top / (mem+1)

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
        return "model_alt"


def matrix_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0.5
    '''
    return 1 / (1 + np.exp(-x))


def dynamic_sigmoid(i, x):
    '''
    Helper sigmoid function where the intercept is a value of i (list)
    '''
    y = np.exp(np.clip(-x, -100, 100)) # avoid runover into infinity.
    out = np.asarray([1 / (1 + ((1 - np.clip(i[j], 1e-50, 1))/np.clip(i[j], 1e-50, 1)) * y[j])
            for j in range(len(i))])
    return out


def stabilize(x):
    '''
    Probably not useful - but keeping it here for now in case it becomes so.
    Yields magnitude value for x between 0 and 1 where the intercept is 0.
    '''
    y = np.array([(-0.5**i)+1 if i > 0 else (-2**i) + 1 for i in x])
    return y
