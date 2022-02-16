'''
General inverse reinforcement learning methodology:
1. guess a reward function
2. compute a policy
3. measure the probability of the given behavior state given the policy
4. gradient on reward function

Our inference:
given a history of states, error, and previous estimate, update estimate of parameters (theta) that determine the 
likelihood of particular qualities of a state.

possible models:
- maximum likelihood
- bayesian
- maximum uniform distribution
- sequence of pdf
- maximum a posteriori

'''


def bayesian_estimate():
    '''
    argmax P_X|Y(-|y)
    '''
