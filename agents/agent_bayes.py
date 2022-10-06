import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import normalize


class Agent_Bayes():
    """
    This agent has an internal model, consisting of a covariance matrix from which it can draw from
    to output behavior and adjust based on errors. An additional matrix determines attention to particular
    features within a given state.

    INPUTS:
        state_size [integer, default=3]: sets size of behavior feature space, N.
        memory [integer >= 0, default=0]: a default minimum memory that gets defaulted to when
        predictions need a large adjustment
        model_var [number, default = 1.]: sets the slope at intercept of the behavior change sigmoid function.
            When set to 0, the model is a matrix of zeroes, meaning behavior does not change from its initial setting.
        behav_initial_spread [number, default = 1.]: multiplier applied within sigmoid to initial behavioral_priors.
            High values create a bimodial distribution. Zero gives 0.5 for all initial behavioral_priors.
        pred_initial_spread [number, default = 1.]: multiplier applied within sigmoid to initial predictions.
                High values create a bimodial distribution. Zero gives 0.5 for all initial predictions.

    VARIABLES:
        b_priors: N length vector, giving probability (to 3 decimal places) of
            agent DISPLAYING features of behavior on trial t.
        behavior: N length vector, indicating presence/absence (1/0) of agent's features of behavior trial t.
        past_priors: list of N length vectors, each recording b_prior for trial t.
        world_pred: N length vector, giving estimated probability (to 3 decimal places)
            of observing features of behavior FROM THE OTHER AGENT on trial t.
    """

    def __init__(self, state_size=3, seed=None, memory=0, behav_control=0, model_var=1, behav_initial_spread=1, pred_initial_spread=1, inference_fn='IRL',  action_cost_fn='linear'):
        assert state_size > 0, "state_size must be > 0"
        self.state_size = state_size  # size of a state
        # generates a new instance of a behavioral prior.
        self.b_priors = np.random.uniform(0, 1, self.state_size)
        # self.behav_initial_spread (>=0) adjusts slope of sigmoid. High values create a bimodal distribution of initial behavioral priors.
        self.behav_initial_spread = behav_initial_spread
        assert behav_initial_spread >= 0, "behav_initial_spread must be >= 0"
        self.b_priors = matrix_sigmoid(
            (self.b_priors)*self.behav_initial_spread)
        self.possible_priors = np.linspace(0, 1, 100)  # stores possible priors
        # alphas and betas for beta distribution to estimate priors of others
        # np.random.normal(2, 4, self.state_size)
        self.alpha = [2]*self.state_size
        self.beta = [2]*self.state_size
        # threshold for re weighting more recent experience
        self.threshold = 0.2
        self.world_pred = [0.5]*state_size
        # self.pred_initial_spread (>=0) adjusts slope of sigmoid. High values create a bimodal distribution of initial behavioral priors.
        self.pred_initial_spread = pred_initial_spread
        assert pred_initial_spread >= 0, "pred_initial_spread must be >= 0"
        # self.world_pred = matrix_sigmoid(
        #    (self.world_pred)*self.pred_initial_spread)
        self.world = []  # history of world states
        assert memory >= 0, "memory must be >= 0"
        self.memory = int(memory)
        self.behav_control = behav_control
        self.past_priors = []  # stores past behavioral priors.
        assert model_var >= 0, "model variance must be at least 0"
        # behavioral model applies some randomness or "personality" to how behavior gets adjusted
        self.model_var = model_var
        self.behav_model = normalize(
            2*np.random.rand(state_size, state_size)-1, axis=1, norm='l2')*self.model_var
        # model_thresh creates distributions where some input changes behavior drastically, while others have small effects.
        # TODO - parameterize this. It's a neat idea, but I don't know how it works yet.
        # self.model_thresh = .95
        # self.behav_model[abs(self.behav_model) > self.model_thresh] = self.behav_model[abs(self.behav_model) > self.model_thresh]*10
        # self.behav_model[abs(self.behav_model) <= self.model_thresh] = self.behav_model[abs(self.behav_model) <= self.model_thresh]*.1

        self.metabolism = 0.0  # metabolic cost so far (accrued via learning)
        self.attn = np.identity(self.state_size)  # attention matrix

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
        #print("bayes pred", self.world_pred)
        #print("alt behavior", self.world[-1])
        if len(self.world[-1]) != len(self.world_pred):
            raise ValueError("state sizes between agents must match")
        dif = self.world[-1] - \
            self.world_pred  # array of differences, for each behavioral feature
        #print("bayes dif", dif)
        avg_abs_error = np.sum(abs(dif))/len(dif)
        #print("bayes avg error", avg_abs_error)
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
        Bayesian posterior estimate of other agent's priors, dictating behviors.
        1. generate priors given all past observed behaviors
        2. generate likelihood of the observations given the prior
        3. get the posterior given the observations
        4. prediction = argmax of the largest prediction
        '''
        self.alpha = [x + y for x, y in zip(self.alpha, self.world[-1])]
        self.beta = [x - y + 1 for x, y in zip(self.beta, self.world[-1])]
        prior_dist = [stats.beta.cdf(x=self.possible_priors+.001, a=self.alpha[i], b=self.beta[i]) - stats.beta.cdf(
            x=self.possible_priors, a=self.alpha[i], b=self.beta[i]) for i in range(self.state_size)]
        likelihood_dist = [stats.binom.pmf(
            k=self.alpha[i], n=len(self.world)+2, p=self.possible_priors) for i in range(self.state_size)]
        posterior_dist = [prior_dist[i]*likelihood_dist[i]
                          for i in range(len(prior_dist))]
        norm_posterior_dist = [p/sum(p) for p in posterior_dist]
        self.world_pred = [np.argmax(norm_posterior_dist[i])/100
                           for i in range(self.state_size)]

    def attention(self):
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
        return "bayes"


def matrix_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0.5
    '''
    return 1 / (1 + np.exp(-1*x))


def dynamic_sigmoid(i, x):
    '''
    Helper sigmoid function where the intercept is a value of i (list)
    '''
    y = np.exp(np.clip(-x, -100, 100))  # avoid runover into infinity.
    out = np.asarray([1 / (1 + ((1 - np.clip(i[j], 1e-50, 1-1e-50))/np.clip(i[j], 1e-50, 1-1e-50)) * y[j])
                      for j in range(len(i))])  # changed clips to keep sigmoid function from getting stuck at 0 or 1
    return out


def stabilize(x):
    '''
    Probably not useful - but keeping it here for now in case it becomes so.
    Yields magnitude value for x between 0 and 1 where the intercept is 0.
    '''
    y = np.array([(-0.5**i)+1 if i > 0 else (-2**i) + 1 for i in x])
    return y
