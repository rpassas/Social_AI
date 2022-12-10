# Social Sandbox

#### Overview
Inspired by research in [social conformity and dynamics](https://www.sciencedirect.com/science/article/abs/pii/S157106452030004X) and [agent-based modeling of ToM and social cooperation](https://arxiv.org/abs/2208.11660), this sandbox is meant to give experimenters a means to explore the mergent dynamics between agents who adjust their behaviors based on the feedback of others. 
<br/>
The core "Sandbox" consists of two main classes: `world` & `agent.` The `world` is the environment that takes in `agents` and can then run an experiments with the agents. It records information, such as behaviors performed by agents, and errors they perceive relative to the behaviors the expected to observe. 
<br/>

#### Running an Experiment
To run an experiment in a notebook, simply import the world class and input experimental parameters you are interested in testing. By design, `world` takes two `agents` who take the parameters given to world. 
<br/>
<br/>
`world = World(state_size =5, time =100, agent=["base", "base"], seed=8)`
<br/>
<br/>
The experimenter can then run run a discrete timestep experiment wherein those two agents interact. At each time step agents generate a behavior, observe their counterpart's behavior, and generally update their internal structures according to prediction error to generate new predictions and behaviors in the next step.
<br/>
<br/>
`world.create_agents()`
<br/>
`world.run()`
<br/>

## Experimental Parameters
Firstly, an experiment runs for a discrete set of timesteps, with each agent having the same number of features in their behaviors. These are determined by the paremters `time` and `state_size` respectively. The `seed` parameter also allows for reproducible experiments, even when randomly initialized.
<br/>

#### Randomly Initialized Parameters
These randomly initialized parameters are the "behavioral priors" and "world predictions." The priors are an array of probabilities (of lenght = `state_size`), determining the likelihood of a behavioral feature at each timestep. Meanwhile, "world predictions" are expectations of future observed behaviors. In addition to a seed, `world` can also take `behav_initial_spread`, which are two values, one for each agent, determining the initial distribution of priors (higher values lead to values of lower variance). Alternatively, experimenters can assign a custom array of priors to each agent via the `init_priors` parameter, assuming that it has a length of `state_size.` The same can be done for world predictiosn using the `pred_initial_spread` `init_preds` parameters.
<br/>

#### Update Function Parameters
Agents adjust both their behavioral priors and predictions with some provided functions (found in `utility.py`). The default is the "sigmoid" function, where error generated (contrast between predictions and behaviors of others), is used to shift priors and/or along a sigmoid curve, centered at the current value (rather than the standard sigmoid curve at 0.5).
<br/>
<br/>
`world = World(state_size =5, time =100, agent=["base", "base"], prediction=["sigmoid", "sigmoid"], behavior=["sigmoid", "sigmoid"])`
<br/>
<br/>
Other functions include stochastic updates, given by the input of "chaos" rather than "sigmoid," wherein updates are arbitrary (positive or negative) but proportional to the error at that timestep. Additionally, the term "orbit" can be given for the `behavior` parameter to allow agents to have periodic behaviors, determined by a transition matrix. While this transition matrix cannot be directly supplied, it is a matrix with complex eigenvalues, and the user can supply a change of basis matrix to customize the end matrix with the `basis_mat` parameter.
<br/>
Experimenters can also adjust the rates at which behaviors and priors are updated, using the `behav_a` and `pred_a` parameters. These values simply act as coefficients to the error before they are used to update any internal structures. Experimenters can also have agents further filter error by using an entropy-based filter to down-weight high-variance behaviors using the `attention` parameter.
<br/>
<br/>
`world = World(state_size =5, time =100, agent=["base", "base"], pred_a = [0.4,0.35], behav_a=[0.1,0.2], prediction=["chaos", "sigmoid"], behavior=["sigmoid", "sigmoid"], attention=["entropy", "entropy"])`
<br/>
<br/>
Agents are also able to estimate how their behaviors change that of others with a simple linear model. They will then enact behaviors to drive counterparts' behavior towards their own predictions. To do so, the user must set the `behav_update` parameter to `True`.

<br/>
#### Getting Experimental Data
Once the experiment is done running, key metrics can be extracted from `world`, including: agent parameters, prediction errors, behaviors, priors, predictability (reflects variance of priors), and others. See the `world` [class](https://github.com/rpassas/Social_AI/blob/main/sandbox/world.py) for all the parameters and methods. 

<br/>
#### PID-PCT Nodes
As an extension of this sandbox, feedback nodes, based on the schemes of [Perceptual Control Theory](https://psycnet.apa.org/record/1974-10192-000) and PID controllers to enact behavior. Again, updates are driven by error signals and the nodes are designed to be hierarchically arranged. There are two main classes: `PCT_node` and `Effector`. The "effector" is the controller as part of the node, both of which can be parameterized as needed.



<br/>
[Demo Notebook](https://github.com/rpassas/Social_AI/blob/main/experiments/base_agent_demo.ipynb)
