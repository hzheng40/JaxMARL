"""
Short introduction to the package.

## Abstract base class
Uses the PettingZoo Parallel API. All agents act synchronously, with actions, 
observations, returns and dones passed as dictionaries keyed by agent names. 
The code can be found in `multiagentgymnax/multi_agent_env.py`

The class follows an identical structure to that of `gymnax` with one execption. 
The class is instatiated with `num_agents`, defining the number of agents within the environment.

## Environment loop
Below is an example of a simple environment loop, using random actions.

"""

import jax 
from multiagentgymnax.mpe import SimpleWorldCommEnv

# Parameters + random keys
max_steps = 100
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Instantiate environment
env = SimpleWorldCommEnv()
obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
print('example action dict', actions)

env.enable_render()

for _ in range(max_steps):
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)
    
    env.render(state)