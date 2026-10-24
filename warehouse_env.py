"""
Warehouse Robot environment: the MDP our agent learns in.

A delivery robot starts at its charging dock (S). It has to drive to the pickup
point (P) to collect a parcel, then deliver the parcel to the packing station (D).

    .  open floor
    #  shelving rack or safety barrier: the robot bumps into it and stays put
    ~  forklift lane: driving into it is a collision, which ends the episode

The agent is never shown this map. It only observes its state and the reward
after each move, so it has to discover a good route by trial and error.
"""

from collections import deque

import numpy as np

LAYOUT = (
    "S.........",
    ".##.##.##.",
    ".##.##.##.",
    ".##.##.##.",
    "..........",
    "#######.##",
    "..........",
    "P........D",
    "~~~~~~~~~~",
)

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
# On a slippery floor the robot can drift sideways instead of moving as intended.
SIDEWAYS = ((2, 3), (2, 3), (0, 1), (0, 1))

# Reward for each kind of step (exactly one of these applies per step).
REWARD_STEP = -1          # ordinary move: costs time and battery, so shorter routes score higher
REWARD_BUMP = -5          # tried to drive into a rack, the barrier or the outer wall
REWARD_PICKUP = 20        # collected the parcel at P (can only happen once per episode)
REWARD_DELIVER = 100      # delivered the parcel at D: success, episode ends
REWARD_COLLISION = -100   # drove into the forklift lane: failure, episode ends


class WarehouseEnv:
    """Grid-world warehouse. State = (row, col, carrying); actions 0-3 = UP, DOWN, LEFT, RIGHT."""

    n_actions = 4

    def __init__(self, layout=LAYOUT, slip_prob=0.0, max_steps=200, seed=None):
        self.layout = layout
        self.n_rows, self.n_cols = len(layout), len(layout[0])
        cells = {(r, c): ch for r, row in enumerate(layout) for c, ch in enumerate(row)}
        self.start = next(p for p, ch in cells.items() if ch == "S")
        self.pickup = next(p for p, ch in cells.items() if ch == "P")
        self.dropoff = next(p for p, ch in cells.items() if ch == "D")
        self.walls = {p for p, ch in cells.items() if ch == "#"}
        self.hazards = {p for p, ch in cells.items() if ch == "~"}
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # Shape of the Q-table without the action axis: rows x cols x (carrying or not).
        self.state_shape = (self.n_rows, self.n_cols, 2)
        self.reset()

    def reset(self):
        """Put the robot back on its dock without a parcel. Returns the start state."""
        self.pos = self.start
        self.carrying = 0
        self.steps = 0
        return self.state

    @property
    def state(self):
        return (self.pos[0], self.pos[1], self.carrying)

    def step(self, action):
        """Try to move the robot one cell.

        Returns (next_state, reward, terminated, truncated, info):
          terminated: the task really ended (parcel delivered, or collision)
          truncated:  we ran out of time (max_steps); the state itself is not terminal
        """
        self.steps += 1
        slipped = self.slip_prob > 0 and self.rng.random() < self.slip_prob
        if slipped:
            action = SIDEWAYS[action][self.rng.integers(2)]
        dr, dc = MOVES[action]
        target = (self.pos[0] + dr, self.pos[1] + dc)

        terminated = False
        if not self._inside(target) or target in self.walls:
            reward, event = REWARD_BUMP, "bump"  # blocked, so the robot stays where it is
        else:
            self.pos = target
            if target in self.hazards:
                reward, event, terminated = REWARD_COLLISION, "collision", True
            elif target == self.pickup and not self.carrying:
                self.carrying = 1
                reward, event = REWARD_PICKUP, "pickup"
            elif target == self.dropoff and self.carrying:
                reward, event, terminated = REWARD_DELIVER, "delivered", True
            else:
                reward, event = REWARD_STEP, "move"

        truncated = not terminated and self.steps >= self.max_steps
        return self.state, reward, terminated, truncated, {"event": event, "slipped": slipped}

    def _inside(self, pos):
        return 0 <= pos[0] < self.n_rows and 0 <= pos[1] < self.n_cols

    def render_ascii(self, state=None):
        """The map as text: r = robot, R = robot carrying the parcel."""
        robot_r, robot_c, carrying = state if state is not None else self.state
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                ch = self.layout[r][c]
                if (r, c) == (robot_r, robot_c):
                    ch = "R" if carrying else "r"
                elif ch == "P" and carrying:
                    ch = "."  # parcel already collected
                row.append(ch)
            lines.append(" ".join(row))
        return "\n".join(lines)

    def shortest_route_length(self):
        """Fewest moves that complete the whole task, found by breadth-first search.

        Only used as the "best possible" reference line on the plots. The agents
        never call this: they have to find the route by themselves.
        """
        start = (*self.start, 0)
        dist = {start: 0}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            r, c, carrying = state
            for dr, dc in MOVES:
                nxt = (r + dr, c + dc)
                if not self._inside(nxt) or nxt in self.walls or nxt in self.hazards:
                    continue
                if carrying and nxt == self.dropoff:
                    return dist[state] + 1
                nxt_state = (*nxt, 1 if nxt == self.pickup else carrying)
                if nxt_state not in dist:
                    dist[nxt_state] = dist[state] + 1
                    queue.append(nxt_state)
        return None


def best_possible_return(route_length):
    """Return of a perfect run: ordinary moves, plus the pickup and delivery steps."""
    return REWARD_STEP * (route_length - 2) + REWARD_PICKUP + REWARD_DELIVER
