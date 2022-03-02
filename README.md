# Social AI

#### Overview

Basic agent experiment where 2 agents behave and try to estimate the other's behavioral priors, using conformity as a way to stabilize the behavior of the other agent (in theory).
<br/>
<br/>
basic_agent.py - class that determines behavior for an agent (no learning or proper estimation)
<br/>
world.py - experiment environment for the agents
<br/>
inference.py - will have the learning and inference models down the line

#### Commands to Run:

`> python world.py -h` for help
<br/>
`> python world.py -b 5 -t 20 -q "chaos" "average" -a 0.3 0.4` for an experiment that runs for 20 timesteps, behavior of size 5, with a chaos and average agent with their respective alphas set to 0.3 and 0.4
<br/>
`> python world.py` to run on default values: 15 timesteps, behavior of size 4, chaos agents, alphas and betas set to 0.5
<br/>
There are other command flags worth playing with once the estimator/learning mechanic is built in
<br/>
<br/>
Flags:
<br/>
-h -> help
<br/>
-s -> behavior size (int)
<br/>
-t -> time steps (int)
<br/>
-q -> agent types (0 or more str)
<br/>
-a -> alphas (0 or more float)
<br/>
-b -> betas (0 or more float)

#### Under the Hood

As of now the agents are initialized on random behavioral priors (between 0 and 1). The expectations of the other's behavior is equally random. At each time step, the agent samples from the priors to generate a behavior. Error is generated be contrasting behaviors and posterior estimations. This error is used to increment metabolic cost and update both the priors of the agents and their estimates of the other's priors. This update is done via a simple heuristic: if estimation error exceeds an arbitrary threshold, update all the estimates and the priors using the error and learning rate. A degree of randomness is incorporated in both updates.
