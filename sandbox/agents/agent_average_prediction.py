import numpy as np


class Agent_Average_Prediction():
    """
    This agent uses memory to average its most recent observed behaviors and adjust its previous
    prediction priors using the average. 

    """
    # TODO: implement inference and cost functions in separate classes
    # inference or parameter estimation should be biased towards most recent states and take error for each state into account

    def __init__(self, state_size=3, alpha=0.5, beta=0.5, seed='', memory=4, inference_fn='IRL',  action_cost_fn='linear'):
        # size of a state
        if state_size < 0:
            self.state_size = 3
        else:
            self.state_size = state_size
        if seed == '':
            seed = np.random.choice(1000, 1)[0]
        np.random.seed(seed)
        # internal model
        self.model = np.random.random((self.state_size, self.state_size))
        # attention model
        self.attention = np.identity(self.state_size)
        # behavioral priors
        self.b_priors = np.random.rand(1, state_size).round(3)[0]
        # current behavior
        self.behavior = []
        # estimate of world state parameters
        self.world_pred = np.random.rand(1, state_size).round(3)[0]
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
        # past predictions
        self.past_predictions = []
        # function for estimating parameters
        self.attn = np.identity(self.state_size)  # attention matrix

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

    def get_predictability(self):
        '''
        Returns the predictability of the agents where a score of 1 indicates priors 
        being close to 0 or 1 (taken as an average across predictability of each prior).
        '''
        predictability = [(abs(p - 0.5))*2 for p in self.b_priors]
        return sum(predictability)/len(predictability)

    def make_prediction(self):
        '''
        Generate actual world prediction (list of 0/1) from priors
        '''
        # return np.random.binomial(1, self.world_pred)
        #p = np.random.binomial(1, self.world_pred)
        self.past_predictions.append(self.world_pred)
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
        '''
        dif = [np.asarray([abs(g-h)
                           for g, h in zip(self.world[-1], self.world_pred)])][0]
        e = round(np.sum(dif)/len(dif), 3)
        return dif, e

    def learn_conform(self):
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust behavioral priors to match the world state based on conformity error
        '''
        dif, pred_error = self.behavior_prediction_error()
        threshold = self.action_cost(pred_error)
        if pred_error > threshold:
            magnitude = np.random.choice([-1, 1]) * pred_error
            r = round(np.random.uniform(0, magnitude), 3) * self.alpha
            self.b_priors = [np.asarray([
                abs(i - r) if abs(i - r) <= 1 else i for i in self.b_priors])][0]
            self.metabolism += pred_error

    def learn_predict_world(self):
        # TODO: this is not learning, just a placeholder heuristic
        # the arbitrary cut off via the action_cost maybe graded rather than all or nothing and still needs to be implemented
        '''
        Adjust prediction of world states based on prediction error.
        Uses alternative weighted average to get vector of errors.
        '''
        history = [0]*self.state_size
        memory = min(len(self.past_predictions), self.memory)
        print(memory)
        for m in range(1, memory+1):
            for p in range(len(self.past_predictions[-m])):
                history[p] = history[p] + (self.past_predictions[-m][p])
        history = [(b+h) for h, b in zip(history, self.world[-1])]
        print(history)
        self.world_pred = [i/(memory+1) for i in history]
        if len(self.past_predictions) < self.memory:
            self.past_predictions[0] = self.world_pred
        print(self.world_pred)
        print(self.world[-1])
        '''
        count = 1
        for i in range(len(self.world)-1, -1, -1):
            # look back four instances
            if count >= self.memory:
                break
            # weight
            #w = 4 - count
            w = 1
            # weighted sum
            for j in range(len(self.past_predictions[i])):
                weighted_history[j] = weighted_history[j] + \
                    (w * self.past_predictions[i][j])
            count += 1
        # divide by sum of weights
        #weighted_history = [i/10 for i in weighted_history]
        weighted_history = [i/count for i in weighted_history]
        # get error vector
        dif, pred_error = self.behavior_prediction_error()
        error = [(p-h)*self.beta for p,
                 h in zip(self.world_pred, weighted_history)]
        # adjust prediction
        new_pred = [(p-e).round(3) for p, e in zip(self.world_pred, error)]
        self.world_pred = new_pred
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
        return "prediction average"

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
