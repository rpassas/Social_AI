import numpy as np
#from utility import matrix_sigmoid
import scipy.stats as stats
from sklearn.preprocessing import normalize
from sandbox.utils.utility import sigmoid_update, chaotic_update, entropy, matrix_sigmoid, linear_update


class Agent():
    """
    The agent class has 

    Args:
        state_size [integer, default=3]: sets size of behavior feature space, N.
        predictions need a large adjustment
        model_var [float, default = 1.]: sets the slope at intercept of the behavior change sigmoid function.
            When set to 0, the model is a matrix of zeros, meaning behavior does not change from its initial setting.
        behav_initial_spread [float, default = 1.]: multiplier applied within sigmoid to initial behavioral_priors.
            High values create a bimodial distribution. Zero gives 0.5 for all initial behavioral_priors.
        pred_initial_spread [float, default = 1.]: multiplier applied within sigmoid to initial predictions.
                High values create a bimodial distribution. Zero gives 0.5 for all initial predictions.

    Attributes:
        b_priors: N length vector, giving probability (to 3 decimal places) of
            agent DISPLAYING features of behavior on trial t.
        behavior: N length vector, indicating presence/absence (1/0) of agent's features of behavior trial t.
        world_pred: N length vector, giving estimated probability (to 3 decimal places)
            of observing features of behavior FROM THE OTHER AGENT on trial t.

    Methods:
        new_behavior(): 
        make_behavior(): 
        get_predictabillity(): 
        make_prediction(): 
        new_behavior(): 
        get_world(world):
        get_behav_priors():
        behavior_prediction_error():
        learn_conform():
        learn_predict_world():
        update_attention():
        get_cost():
        get_attention():
        get_costs():
        get_avg_costs():
        get_behav_model():
        get_type():
    """

    def __init__(self, state_size=3, seed=None,  model_var=1, behav_initial_spread=1, pred_initial_spread=1, pred_a=0, behav_a=0, prediction='sigmoid', behavior='sigmoid', attention='static'):
        self.flag = True
        if seed:
            np.random.seed(seed)
        assert state_size > 0, "state_size must be > 0"
        self.state_size = state_size  # size of a state
        # generates a new instance of a behavioral prior.
        self.b_priors = np.random.normal(0, 1, self.state_size)
        # self.behav_initial_spread (>=0) adjusts slope of sigmoid.
        # High values create a bimodal distribution of initial behavioral priors.
        self.behav_initial_spread = behav_initial_spread
        assert behav_initial_spread >= 0, "behav_initial_spread must be >= 0"
        self.b_priors = matrix_sigmoid(
            (self.b_priors)*self.behav_initial_spread)
        self.world_pred = np.random.normal(0, 1, self.state_size)
        # self.pred_initial_spread (>=0) adjusts slope of sigmoid. High values create a bimodal distribution of initial behavioral priors.
        self.pred_initial_spread = pred_initial_spread
        assert pred_initial_spread >= 0, "pred_initial_spread must be >= 0"
        self.world_pred = matrix_sigmoid(
            (self.world_pred)*self.pred_initial_spread)
        self.world = []  # history of world states
        # for recordin the most recent behavior
        self.current_behavior = []
        self.past_behavior = []
        assert model_var >= 0, "model variance must be at least 0"
        # pred_a is the learning rate for adjusting the agent's prediction or reference signal
        assert pred_a >= 0, "model variance must be at least 0"
        assert pred_a <= 1, "model variance must be at most 1"
        self.pred_a = pred_a
        # behav_a is the learning rate for adjusting the agent's beahvioral model
        assert behav_a >= 0, "model variance must be at least 0"
        assert behav_a <= 1, "model variance must be at most 1"
        self.behav_a = behav_a
        self.model_var = model_var
        # behavioral model applies some randomness or "personality" to how behavior gets adjusted
        #self.behav_model = np.random.uniform(0, 1, self.state_size)
        self.behav_model = np.random.rand(self.state_size, 2)
        # seeks to estimate the behav_model and prior of others
        self.model_estimate = np.ones((self.state_size, 2))
        x_vals = np.array([-0.5, 0, 0.5])
        dim_0 = np.ones(x_vals.shape[0])
        x = np.c_[dim_0, x_vals]
        self.x = np.tile(x, (self.state_size, 1, 1))
        self.b_count = np.zeros((state_size, 3))
        self.obs_sum = np.zeros((state_size, 3))
        self.y = np.zeros((state_size, 3))
        # self.behav_model = normalize(
        #    2*np.random.rand(state_size, state_size)-1, axis=1, norm='l2')*self.model_var
        # model_thresh creates distributions where some input changes behavior drastically, while others have small effects.
        # TODO - parameterize this. It's a neat idea, but I don't know how it works yet.
        # self.model_thresh = .95
        # self.behav_model[abs(self.behav_model) > self.model_thresh] = self.behav_model[abs(self.behav_model) > self.model_thresh]*10
        # self.behav_model[abs(self.behav_model) <= self.model_thresh] = self.behav_model[abs(self.behav_model) <= self.model_thresh]*.1

        self.metabolism = 0.0  # metabolic cost so far (accrued via learning)
        self.attn = np.identity(self.state_size)  # attention matrix
        self.pred_func = prediction
        self.behav_func = behavior
        self.attn_func = attention

    def new_behavior(self):
        '''
        Create new random behavior, initialized as the first behavioral prior was.
        '''
        self.b_priors = np.random.normal(0, 1, self.state_size)
        self.b_priors = matrix_sigmoid(
            (2*self.b_priors-1)*self.behav_initial_spread)

    def make_behavior(self):
        '''
        Generate actual behavior (list of 0/1) from priors
        '''
        self.past_behavior = self.current_behavior
        self.current_behavior = np.random.binomial(1, self.b_priors)
        return self.current_behavior

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
        return self.world_pred

    def get_world(self, world):
        '''
        For adding current world state (i.e. other agent's behavior) to the history
        '''
        for b in range(len(self.past_behavior)):
            if self.past_behavior[b] == 0:
                self.obs_sum[b][0] += world[b]
                self.b_count[b][0] += 1
            elif self.past_behavior[b] == 1:
                self.obs_sum[b][2] += world[b]
                self.b_count[b][2] += 1
            self.b_count[b][1] += 1
            self.obs_sum[b][1] += world[b]
        if len(self.past_behavior) == 0:
            for s in range(self.state_size):
                self.b_count[s][1] += 1
                self.obs_sum[s][1] = world[s]

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
        avg_abs_error = np.sum(abs(dif))/len(dif)
        return dif, avg_abs_error

    def learn_conform(self):
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        #updated_dif = self.behav_model @ attn_weighted_dif
        if len(self.past_behavior) > 0:
            self.update_behav_model()
        updated_dif = self.behav_model.T[1] * attn_weighted_dif
        if self.behav_func == 'static':
            pass
        elif self.behav_func == 'chaos':
            self.b_priors = chaotic_update(
                self.b_priors, 0.2, avg_abs_error)
        elif self.behav_func == 'linear':
            self.b_priors = linear_update(self.b_priors, updated_dif)
        elif self.behav_func == 'sigmoid':
            self.b_priors = sigmoid_update(
                center=self.b_priors, error=updated_dif)
        else:
            raise ValueError("Behavioral update parameter invalid")

    def learn_predict_world(self):
        dif, avg_abs_error = self.behavior_prediction_error()
        attn_weighted_dif = self.attn @ dif
        updated_dif = self.model_estimate.T[1] * attn_weighted_dif
        if self.pred_func == 'static':
            pass
        elif self.pred_func == 'chaos':
            self.world_pred = chaotic_update(
                self.world_pred, 0.2, avg_abs_error)
        elif self.behav_func == 'linear':
            self.world_pred = linear_update(
                self.world_pred, attn_weighted_dif, self.pred_a)
        elif self.pred_func == 'sigmoid':
            self.world_pred = sigmoid_update(
                center=self.world_pred, error=self.pred_a*updated_dif)
        else:
            raise ValueError("Prediction update parameter invalid")

    # def update_model(self):

    def update_behav_model(self):
        for s in range(self.state_size):
            self.y[s][0] = self.obs_sum[s][0] / max(self.b_count[s][0], 0.001)
            self.y[s][1] = self.obs_sum[s][1]/self.b_count[s][1]
            self.y[s][2] = self.obs_sum[s][2] / max(self.b_count[s][2], 0.001)
        for m in range(len(self.behav_model)):
            hypothesis = np.dot(self.x[m], self.behav_model[m])
            loss = hypothesis - self.y[m]
            grad = np.dot(self.x[m].T, loss) / self.x[m].shape[0]
            self.behav_model[m] = self.behav_model[m] - grad*self.behav_a

    def update_attention(self):
        if self.attn_func == 'static':
            pass
        elif self.attn_func == 'entropy':
            self.attn = entropy(prob=self.world_pred)
        else:
            raise ValueError("Attention update parameter invalid")

    def get_cost(self):
        '''
        Get the cost so far.
        '''
        return self.metabolism

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

    def get_behav_model(self):
        return self.behav_model

    def get_type(self):
        '''
        Get the agent type.
        '''
        return f"p: {self.pred_func}, b: {self.behav_func}, attn: {self.attn_func}, pred a: {self.pred_a}, behav a: {self.behav_a} "
