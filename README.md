# Social AI

#### Overview

Basic agent experiment where 2 agents behave and try to estimate the other's behavioral priors, using conformity as a way to stabilize the behavior of the other agent (in theory).
<br/>
<br/>
agent.py - class that determines behavior for an agent
<br/>
world.py - experiment environment for the agents
<br/>
inference.py - will have the learning and inference models down the line

#### Commands to Run:

`> python world.py -h` for help
<br/>
`> python world.py -b 5 -t 20` for an experiment that runs for 20 timesteps, behavior of size 5
<br/>
`> python world.py` to run on default values: 15 timesteps, behavior of size 4

#### Under the Hood

As of now the agents are initialized on random behavioral priors (between 0 and 1). The expectations of the other's behavior is initialized to be the same. At each time step, the agent samples from the priors to generate a behavior. Error is generated be contrasting behaviors and posterior estimations. This error is used to increment metabolic cost and update both the priors of the agents and their estimates of the other's priors. This update is done via a simple heuristic: if error exceeds 50%, update all the priors and estimates by subtracting a random number between 0 and the error (absolute value of the difference is kept as the new value).
