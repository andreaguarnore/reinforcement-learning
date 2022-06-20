__all__ = [
    'generate_episode',
]


from gym import Env


def generate_episode(
    env: Env,
    sample_function: callable,
    sample_function_args: dict,
    n_steps: int | None,
) -> list[tuple[int, int, float]]:
    """
    Generate an episode following the given policy.
    """
    episode = []
    state = env.reset()
    step = 0
    while n_steps is None or step < n_steps:
        action = sample_function(state, **sample_function_args)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
        step += 1
    return episode
