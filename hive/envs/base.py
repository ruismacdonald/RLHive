from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Base class for environments, the learning task e.g. an MDP.
    """

    def __init__(self):
        self._env_spec = None

    @abstractmethod
    def reset(self):
        """
        Resets the state of the environment.

        Returns:
            observation: The initial observation of the new episode.
            turn (int): The index of the agent which should take turn.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Run one time-step of the environment using the input action.

        Args:
            action: An element of environment's action space.

        Returns:
            observation: Indicates the next state that is an element of environment's observation space.
            reward (float): A scalar reward achieved from the transition.
            done (bool): Indicates whether the episode has ended.
            turn: Indicates which agent should take turn.
            info (dict): Additional custom information.
        """

        raise NotImplementedError

    def render(self, mode='rgb_array'):
        """
        Displays a rendered frame from the environment.
        """
        pass

    @abstractmethod
    def seed(self, seed=None):
        """
        Reseeds the environment.
        """
        pass

    def get_random_observation(self):
        """
        Returns a valid random observation of the environment.

        Returns:
            observation: A random observation that is an element of environment's observation space.
            turn (int): The index of the agent which should take turn.
        """
        pass

    def set_observation(self, observation, turn=None):
        """
        Changes the observation and turn in the environment to the input values.
        """
        pass

    def save(self, save_dir):
        """
        Saves the environment.
        """
        pass

    def load(self, load_dir):
        """
        Loads the environment.
        """
        pass

    def close(self):
        """
        Additional clean up operations
        """
        pass

    @property
    def env_specs(self):
        return self._env_spec

    @env_specs.setter
    def env_specs(self, env_spec):
        self._env_spec = env_spec
