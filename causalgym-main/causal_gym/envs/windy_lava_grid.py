from __future__ import annotations

import numpy as np
import pygame
from gymnasium import spaces

from causal_gym import SCM


class WindyLavaGridEnv(SCM):
    """Windy gridworld variant with configurable lava cells and wind patterns."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        grid_shape: tuple[int, int] = (12, 4),
        start_location: tuple[int, int] | None = None,
        target_location: tuple[int, int] | None = None,
        lava_cells: list[tuple[int, int]] | None = None,
        wind_strength: tuple[float, float, float, float, float] = (0.6, 0.1, 0.1, 0.1, 0.1),
        column_wind: dict[int, int] | None = None,
        step_penalty: float = -1.0,
        lava_penalty: float = -20.0,
        goal_reward: float = 0.0,
        terminate_on_lava: bool = True,
        behavior_policy=None,
    ):
        self.width, self.height = grid_shape
        self.window_size = 512
        self._max_indices = np.array([self.width - 1, self.height - 1], dtype=int)

        # Observations mirror WindyGridWorld so notebooks can be reused.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.zeros(2, dtype=int), high=self._max_indices, dtype=int),
                "target": spaces.Box(low=np.zeros(2, dtype=int), high=self._max_indices, dtype=int),
                "wind": spaces.Discrete(5),
            }
        )
        self.action_space = spaces.Discrete(5)
        self.wind_space = np.array([0, 1, 2, 3, 4], dtype=int)

        self.wind_strength = np.asarray(wind_strength, dtype=float)
        self.wind_strength = self.wind_strength / self.wind_strength.sum()

        self._wind_to_direction = {
            0: np.array([0, 0], dtype=int),   # calm
            1: np.array([1, 0], dtype=int),   # push right
            2: np.array([0, 1], dtype=int),   # push down
            3: np.array([-1, 0], dtype=int),  # push left
            4: np.array([0, -1], dtype=int),  # push up
        }
        self._action_to_direction = {
            0: np.array([0, 0], dtype=int),   # stay
            1: np.array([1, 0], dtype=int),   # right
            2: np.array([0, 1], dtype=int),   # down
            3: np.array([-1, 0], dtype=int),  # left
            4: np.array([0, -1], dtype=int),  # up
        }

        self._policy = behavior_policy
        self.step_penalty = step_penalty
        self.lava_penalty = lava_penalty
        self.goal_reward = goal_reward
        self.terminate_on_lava = terminate_on_lava

        self._start_location = np.array(start_location or (0, self.height - 1), dtype=int)
        self._target_default = np.array(target_location or (self.width - 1, self.height - 1), dtype=int)

        if lava_cells is None:
            lava_cells = [(col, self.height - 1) for col in range(1, self.width - 1)]
        self._lava_cells = {tuple(cell) for cell in lava_cells}

        for coordinate in list(self._lava_cells) + [tuple(self._start_location), tuple(self._target_default)]:
            if not self._within_bounds(coordinate):
                raise ValueError(f"Coordinate {coordinate} is outside the grid bounds {grid_shape}.")
        if tuple(self._start_location) in self._lava_cells:
            raise ValueError("Start location cannot lie inside lava.")
        if tuple(self._target_default) in self._lava_cells:
            raise ValueError("Target location cannot lie inside lava.")

        self.column_wind = column_wind or {
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 1,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._agent_location = self._start_location.copy()
        self._target_location = self._target_default.copy()
        self._wind_direction = 0
        self.steps_taken = 0

    def _within_bounds(self, coordinate: tuple[int, int]) -> bool:
        return 0 <= coordinate[0] < self.width and 0 <= coordinate[1] < self.height

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "target": self._target_location.copy(), "wind": int(self._wind_direction)}

    def _get_info(self, lava_hit: bool = False, goal_reached: bool = False):
        info = {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "lava_hit": lava_hit,
            "success": goal_reached and not lava_hit,
            "steps": self.steps_taken,
        }
        current_column = int(self._agent_location[0])
        if current_column in self.column_wind:
            info["column_wind"] = self.column_wind[current_column]
        return info

    def action(self):
        if callable(self._policy):
            return self._policy(self._get_obs())
        if isinstance(self._policy, np.ndarray):
            x_idx, y_idx = map(int, self._agent_location)
            return int(self._policy[x_idx, y_idx, self._wind_direction])
        if self._policy is not None:
            return self._policy
        return self.action_space.sample()

    def observation(self):
        return self._get_obs()

    def see(self):
        action = self.action()
        observation, reward, terminated, truncated, info = self.step(action)
        return action, observation, reward, terminated, truncated, info

    def do(self, action):
        observation, reward, terminated, truncated, info = self.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, agent_location=None):
        super().reset(seed=seed)
        options = options or {}

        target = options.get("target_location", self._target_default)
        start = agent_location or options.get("start_location", self._start_location)

        self._target_location = np.array(target, dtype=int)
        self._agent_location = np.array(start, dtype=int)

        if tuple(self._target_location) in self._lava_cells:
            raise ValueError("Target cannot be placed on lava.")
        if tuple(self._agent_location) in self._lava_cells:
            raise ValueError("Agent cannot start on lava.")

        self._wind_direction = np.random.choice(self.wind_space, 1, p=self.wind_strength)[0]
        self.steps_taken = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        random_wind = self._wind_to_direction[int(self._wind_direction)]

        column_push = np.array([0, 0], dtype=int)
        if int(self._agent_location[0]) in self.column_wind:
            strength = self.column_wind[int(self._agent_location[0])]
            column_push = np.array([0, -strength], dtype=int)

        proposed_location = self._agent_location + direction + random_wind + column_push
        self._agent_location = np.clip(proposed_location, np.zeros(2, dtype=int), self._max_indices)
        self.steps_taken += 1

        lava_hit = tuple(self._agent_location) in self._lava_cells
        goal_reached = np.array_equal(self._agent_location, self._target_location)

        terminated = bool(goal_reached or (self.terminate_on_lava and lava_hit))
        truncated = False

        if goal_reached:
            reward = self.goal_reward
        elif lava_hit:
            reward = self.lava_penalty
        else:
            reward = self.step_penalty

        info = self._get_info(lava_hit=lava_hit, goal_reached=goal_reached)

        self._wind_direction = np.random.choice(self.wind_space, 1, p=self.wind_strength)[0]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        offset = 50

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size + 2 * offset, self.window_size + 2 * offset))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size + 2 * offset, self.window_size + 2 * offset))
        canvas.fill((0, 0, 0))

        pix_square_size_x = self.window_size / self.width
        pix_square_size_y = self.window_size / self.height

        for (x_cell, y_cell) in self._lava_cells:
            pygame.draw.rect(
                canvas,
                (178, 34, 34),
                pygame.Rect(
                    np.array([x_cell * pix_square_size_x, y_cell * pix_square_size_y]) + offset,
                    (pix_square_size_x, pix_square_size_y),
                ),
            )

        pygame.draw.rect(
            canvas,
            (50, 205, 50),
            pygame.Rect(
                self._target_location * np.array([pix_square_size_x, pix_square_size_y]) + offset,
                (pix_square_size_x, pix_square_size_y),
            ),
        )

        pygame.draw.circle(
            canvas,
            (255, 165, 0),
            (self._agent_location + 0.5) * np.array([pix_square_size_x, pix_square_size_y]) + offset,
            min(pix_square_size_x, pix_square_size_y) / 3,
        )

        for idx in range(self.width + 1):
            pygame.draw.line(
                canvas,
                (112, 112, 112),
                (offset + idx * pix_square_size_x, offset),
                (offset + idx * pix_square_size_x, self.window_size + offset),
                width=2,
            )

        for idx in range(self.height + 1):
            pygame.draw.line(
                canvas,
                (112, 112, 112),
                (offset, offset + idx * pix_square_size_y),
                (self.window_size + offset, offset + idx * pix_square_size_y),
                width=2,
            )

        border_rects = [
            pygame.Rect(np.array([0, 0], dtype=int), (self.window_size + 2 * offset, offset)),
            pygame.Rect(np.array([0, 0], dtype=int), (offset, self.window_size + 2 * offset)),
            pygame.Rect(np.array([self.window_size + offset, 0], dtype=int), (offset, self.window_size + 2 * offset)),
            pygame.Rect(np.array([0, self.window_size + offset], dtype=int), (self.window_size + 2 * offset, offset)),
        ]
        for rect in border_rects:
            pygame.draw.rect(canvas, (112, 112, 112), rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
