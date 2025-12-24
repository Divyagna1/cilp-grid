from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


PolicyFn = Callable[[Dict[str, Any], Dict[str, Any], Any], Any]


def _to_serializable(value: Any) -> Any:
    """Convert numpy-heavy structures into JSON-friendly objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(val) for val in value]
    return value


def _copy_obs(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {
            key: np.array(val, copy=True) if isinstance(val, np.ndarray) else val
            for key, val in obs.items()
        }
    if isinstance(obs, np.ndarray):
        return np.array(obs, copy=True)
    return obs


def _copy_info(info: Any) -> Any:
    if info is None:
        return None
    if isinstance(info, dict):
        return {
            key: np.array(val, copy=True) if isinstance(val, np.ndarray) else val
            for key, val in info.items()
        }
    return info


@dataclass
class Transition:
    observation: Dict[str, Any]
    action: Any
    reward: float
    next_observation: Dict[str, Any]
    info: Dict[str, Any]
    terminated: bool
    truncated: bool
    step_index: int

    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": _to_serializable(self.observation),
            "next_observation": _to_serializable(self.next_observation),
            "action": _to_serializable(self.action),
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "info": _to_serializable(self.info),
            "step_index": int(self.step_index),
        }


@dataclass
class Trajectory:
    transitions: List[Transition] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    final_info: Dict[str, Any] = field(default_factory=dict)

    def length(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "total_reward": float(self.total_reward),
            "final_info": _to_serializable(self.final_info),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


class TrajectoryCollector:
    """Roll out SCM/Gym environments and keep both successful and failed traces."""

    def __init__(
        self,
        env: Any,
        policy: Optional[PolicyFn] = None,
        max_steps: int = 200,
        success_criterion: Optional[Callable[[bool, bool, Dict[str, Any]], bool]] = None,
        use_behavior_policy: bool = False,
    ):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.use_behavior_policy = use_behavior_policy and policy is None
        self.success_criterion = success_criterion or self._default_success

    def collect(self, episodes: int, reset_kwargs: Optional[Dict[str, Any]] = None) -> List[Trajectory]:
        traces: List[Trajectory] = []
        for _ in range(episodes):
            traces.append(self.rollout(reset_kwargs=reset_kwargs or {}))
        return traces

    def rollout(self, reset_kwargs: Optional[Dict[str, Any]] = None) -> Trajectory:
        reset_kwargs = reset_kwargs or {}
        observation, info = self.env.reset(**reset_kwargs)
        transitions: List[Transition] = []
        total_reward = 0.0
        final_info: Dict[str, Any] = {}

        for step in range(self.max_steps):
            if self.use_behavior_policy:
                action, next_observation, reward, terminated, truncated, step_info = self._step_with_behavior_policy()
            else:
                action = self._select_action(observation, info)
                next_observation, reward, terminated, truncated, step_info = self._step_with_action(action)

            transitions.append(
                Transition(
                    observation=_copy_obs(observation),
                    action=action,
                    reward=float(reward),
                    next_observation=_copy_obs(next_observation),
                    info=_copy_info(step_info),
                    terminated=terminated,
                    truncated=truncated,
                    step_index=step,
                )
            )

            total_reward += float(reward)
            final_info = _copy_info(step_info)
            observation = next_observation
            info = step_info

            if terminated or truncated:
                break

        success = self.success_criterion(transitions[-1].terminated if transitions else False,
                                         transitions[-1].truncated if transitions else False,
                                         final_info)

        return Trajectory(
            transitions=transitions,
            success=success,
            total_reward=total_reward,
            final_info=final_info,
        )

    def partition(self, trajectories: Sequence[Trajectory]) -> Dict[str, List[Trajectory]]:
        successful = [traj for traj in trajectories if traj.success]
        failed = [traj for traj in trajectories if not traj.success]
        return {"success": successful, "failure": failed}

    def save(self, trajectories: Sequence[Trajectory], path: str | Path):
        payload = [trajectory.to_dict() for trajectory in trajectories]
        Path(path).write_text(json.dumps(payload, indent=2))

    def _select_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Any:
        if callable(self.policy):
            return self.policy(observation, info, self.env)
        if isinstance(self.policy, np.ndarray):
            agent_row, agent_col = observation["agent"]
            wind = observation.get("wind", 0)
            return int(self.policy[agent_row, agent_col, wind])
        if self.policy is not None:
            return self.policy
        if hasattr(self.env, "action"):
            return self.env.action()
        return self.env.action_space.sample()

    def _step_with_behavior_policy(self):
        outcome = self.env.see()
        if len(outcome) == 6:
            action, next_obs, reward, done_flag, aux_flag, info = outcome
            terminated = bool(done_flag)
            truncated = bool(aux_flag)
        elif len(outcome) == 5:
            action, next_obs, reward, done_flag, info = outcome
            terminated = bool(done_flag)
            truncated = False
        else:
            raise ValueError("Unexpected number of outputs from env.see()")
        return action, next_obs, reward, terminated, truncated, info

    def _step_with_action(self, action: Any):
        if hasattr(self.env, "do"):
            outcome = self.env.do(action)
        else:
            outcome = self.env.step(action)

        if len(outcome) == 5:
            next_obs, reward, done_flag, aux_flag, info = outcome
            terminated = bool(done_flag)
            truncated = bool(aux_flag)
        elif len(outcome) == 4:
            next_obs, reward, done_flag, info = outcome
            terminated = bool(done_flag)
            truncated = False
        else:
            raise ValueError("Unexpected number of outputs from env.do/step()")

        return next_obs, reward, terminated, truncated, info

    @staticmethod
    def _default_success(terminated: bool, truncated: bool, info: Dict[str, Any]) -> bool:
        info = info or {}
        if "success" in info:
            return bool(info["success"])
        return bool(terminated and not truncated)


def save_trajectories(trajectories: Iterable[Trajectory], path: str | Path):
    payload = [trajectory.to_dict() for trajectory in trajectories]
    Path(path).write_text(json.dumps(payload, indent=2))
