import jax
import jax.numpy as jnp

def to_common_obs(original_obs: jax.Array, original_type: str):
    """
    Wrap original observation to common observation

    Common observations: [self_vel, self_pos, closest_other_agent, closest_landmark]
    """
    pass