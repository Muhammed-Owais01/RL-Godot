from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from godot_rl.core.godot_env import GodotEnv


class GodotVecEnv(VecEnv):
    def __init__(
        self,
        ports: List[int],
        env_path: Optional[str] = None,
        seed: int = 0,
        show_window: bool = False,
        framerate: Optional[int] = None,
        action_repeat: Optional[int] = None,
        speedup: Optional[int] = None,
    ) -> None:
        self.envs = [
            GodotEnv(
                env_path=env_path,
                port=port,
                seed=seed + i,
                show_window=show_window,
                framerate=framerate,
                action_repeat=action_repeat,
                speedup=speedup,
                convert_action_space=True,
            )
            for i, port in enumerate(ports)
        ]

        self.n_parallel = len(self.envs)
        self._results = None

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def num_envs(self) -> int:
        return self.envs[0].num_envs * self.n_parallel

    def reset(self) -> Dict[str, np.ndarray]:
        all_obs = []
        for env in self.envs:
            obs, _ = env.reset()
            all_obs.extend(obs)
        return self._stack_obs(all_obs)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        num_envs = self.envs[0].num_envs

        for i, env in enumerate(self.envs):
            env_actions = actions[i * num_envs : (i + 1) * num_envs]
            env.step_send(env_actions)

        all_obs = []
        all_rewards = []
        all_dones = []
        all_info: List[Dict[str, Any]] = []

        for env in self.envs:
            obs, reward, term, trunc, info = env.step_recv()
            all_obs.extend(obs)
            all_rewards.extend(reward)
            all_dones.extend(term)
            all_info.extend(info)

        return self._stack_obs(all_obs), np.array(all_rewards, dtype=np.float32), np.array(all_dones), all_info

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def step_async(self, actions: np.ndarray) -> None:
        self._results = self.step(actions)

    def step_wait(self):
        return self._results

    def env_is_wrapped(self, wrapper_class: type, indices: Optional[List[int]] = None) -> List[bool]:
        return [False] * self.num_envs

    def env_method(self, *args, **kwargs):
        raise NotImplementedError()

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        if attr_name == "render_mode":
            return [None for _ in range(self.num_envs)]
        raise AttributeError("get_attr not implemented")

    def seed(self, seed=None):
        raise NotImplementedError()

    def set_attr(self, *args, **kwargs):
        raise NotImplementedError()

    def _stack_obs(self, all_obs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        keys = all_obs[0].keys()
        return {key: np.array([o[key] for o in all_obs]) for key in keys}
