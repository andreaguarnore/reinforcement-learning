__all__ = [
    'Logger',
]


import logging
from logging import FileHandler
from logging import Formatter


class Logger:

    def __init__(self, logger_name: str, filename: str = None, tab_size: int = 3) -> None:
        log_format = '%(message)s'
        log_level = logging.INFO

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Stream handler
        if filename is None:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(Formatter(log_format))
            self.logger.addHandler(stream_handler)

        # File handler
        else:
            file_handler = FileHandler(filename, mode='w')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(Formatter(log_format))
            self.logger.addHandler(file_handler)

        self.tab = ' ' * tab_size
        self.episode = 0

    def log_training_episode(self, episode: int, n_steps: int, total_reward: float) -> None:
        """
        Write to `stdout` info regarding the last training episode.
        """
        self.logger.info(f'episode {episode}')
        self.logger.info(f'{self.tab}steps: {n_steps}')
        self.logger.info(f'{self.tab}total reward: {total_reward:.2f}\n')

    def save_episode_step(self, state: int | float, action: int, reward: float) -> None:
        """
        Save to file a step of an episode.
        """
        self.logger.info(f'{self.episode} {state} {action} {reward}')

    def save_episode(self, episode) -> None:
        """
        Save to file all steps of an episode.
        """
        for state, action, reward in episode:
            self.save_episode_step(state, action, reward)

    def new_episode(self) -> None:
        self.episode += 1
