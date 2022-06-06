

def error_convergence(e):
    '''
    Calculates the change in error during an experiment.
    '''
    return [abs(e[i-1]-e[i]) for i in range(1, len(e))]


def convergence_test(x, epsilon=0.1, var=.8):
    '''
    takes:
    x         a list of values
    epsilon   a threshold of error change
    var       a proportion of transitions that must remain within

    An asymptote means that a proportion (var) of steps are less than
    the threshold (epsilon). Asymptotes must last at least 10% of timesteps 

    returns steps to convergence & plateau
    '''
    # when do we hit an asymptote?
    times_converged = []
    # where does error locally plateau?
    asymptotes = []
    # how long must an asymptote be?
    a_len = int(len(x) * .1)
    # are we on an asymptote?
    on_asymptote = False
    # start of an asymptote
    a_start = 0
    # record of passed threshold tests
    c = []
    # check difference in error for each timestep
    for i in range(len(x)-1):
        j = i+1
        # error improved less than the threshold
        if x[j] - x[i] < epsilon:
            # current timestep passed
            c.append(1)
            if not on_asymptote:
                if j >= a_len:
                    # if we are not on an asymptote and not just starting, increment start
                    a_start += 1
                # verify if we are on asymptote
                on_asymptote = check_convergence(c[a_start:j], var)
            elif not check_convergence(c[a_start: j], var):
                #
                if j >= a_len:
                    # no longer on asymptote
                    times_converged.append(a_start)
                    e_avg = sum(x[a_start:j])/len(x[a_start:j])
                    asymptotes.append(e_avg)
                    a_start += 1
        else:
            # current timestep failed
            c.append(0)
            on_asymptote = check_convergence(c[a_start:j], var)
            if not on_asymptote and j >= a_len:
                # increment start if not at beginning
                a_start += 1

    return times_converged, asymptotes


def check_convergence(c, var):
    '''
    takes:
    c      list of binary values indicating error below threshold
    var    proporiton of values that need to be below the threshold

    returns boolean indicating convergence
    '''
    avg_c = sum(c)/len(c)
    return avg_c >= var
