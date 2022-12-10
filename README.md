# Social Sandbox

#### Overview
Inspired by research in [social conformity and dynamics](https://www.sciencedirect.com/science/article/abs/pii/S157106452030004X) and [agent-based modeling of ToM and social cooperation](https://arxiv.org/abs/2208.11660), this sandbox is meant to give experimenters a means to explore the mergent dynamics between agents who adjust their behaviors based on the feedback of others. 
<br/>
The core "Sandbox" consists of two main classes: `world` & `agent.` The `world` is the environment that takes in `agents` and can then run an experiments with the agents. It records information, such as behaviors performed by agents, and errors they perceive relative to the behaviors the expected to observe. 
<br/>

#### Running an Experiment
To run an experiment in a notebook, simply import the world class and input experimental parameters you are interested in testing. By design, `world` takes two `agents` who take the parameters given to world.
<br/>
`world = World(state_size =5, time =100, agent=["base", "base"], seed=8)`
<br/>
The experimenter can then run run a discrete timestep experiment wherein those two agents interact.
<br/>
`world.create_agents()`
<br/>
`world.run()`
<br/>

#### Experimental Parameters
Firstly, an experiment runs for a discrete set of timesteps, with each agent having the same number of features in their behaviors. These are determined by the paremters `time` and `state_size` respectively. The `seed` parameter also allows for reproducible experiments, even when randomly initialized.
<br/>
These randomly initialized parameters are the "behavioral priors" and "world predictions." The priors are an array of probabilities (of lenght = `state_size`), determining the likelihood of a behavioral feature at each timestep. Meanwhile, "world predictions" are expectations of future observed behaviors. In addition to a seed, `world` can also take `behav_initial_spread`, which are two values, one for each agent, determining the initial distribution of priors (higher values lead to values of lower variance). Alternatively, experimenters can assign a custom array of priors to each agent via the `init_priors` parameter, assuming that it has a length of `state_size.` The same can be done for world predictiosn using the `pred_initial_spread` `init_preds` parameters.
<br/>





