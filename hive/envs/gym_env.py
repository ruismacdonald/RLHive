import gym

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec


class GymEnv(BaseEnv):
    """
    Gym 0.26.2-compatible wrapper with explicit seeding.
    """

    def __init__(self, env_name, num_players=1, **kwargs):
        """
        Args:
            env_name (str): Name of the environment (NOTE: make sure it is available
                in gym.envs.registry.all())
            num_players (int): Number of players for the environment.
            kwargs: Any arguments you want to pass to :py:meth:`create_env` or
                :py:meth:`create_env_spec` can be passed as keyword arguments to this
                constructor.
        """
        self._seed = None
        self.create_env(env_name, **kwargs)
        super().__init__(self.create_env_spec(env_name, **kwargs), num_players)
        
    def set_seed(self, seed: int):
        """Call this right after creating the env (from your runner)."""
        self._seed = int(seed)
        try: self._env.action_space.seed(self._seed)
        except Exception: pass
        try: self._env.observation_space.seed(self._seed)
        except Exception: pass
    
    # Satisfy BaseEnv's abstract method
    def seed(self, seed=None):
        self.set_seed(seed)
        return [seed]

    def create_env(self, env_name, **kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        self._env = gym.make(env_name)

    def create_env_spec(self, env_name, **kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            observation_spaces = self._env.observation_space.spaces
        else:
            observation_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            action_spaces = self._env.action_space.spaces
        else:
            action_spaces = [self._env.action_space]

        return EnvSpec(
            env_name=env_name,
            observation_space=observation_spaces,
            action_space=action_spaces,
        )
    
    def reset(self, seed=None):
        if seed is None:
            seed = self._seed

        try:
            # Gym >=0.26 API
            obs, info = self._env.reset(seed=seed)
            return obs, self._turn
        except TypeError:
            # Gym <=0.25 API: must call .seed() separately
            if seed is not None:
                self._env.seed(seed)
            obs = self._env.reset()
            return obs, self._turn

    def step(self, action):
        step_result = self._env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated 
        elif len(step_result) == 4:
            observation, reward, done, info = step_result
        else:
            raise ValueError(f"Unexpected step return length: {len(step_result)}")
    
        self._turn = (self._turn + 1) % self._num_players
        return observation, reward, done, self._turn, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)
        
    def close(self):
        self._env.close()
