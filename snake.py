"""Snake game env + dataset generator.

Output frame is always 64x64 RGB. The grid_cells parameter sets how many cells
fit per side: grid_cells=64 → each cell is 1px (current default), grid_cells=16
→ each cell is 4px (chunky, easier to learn).

Actions: 0=up, 1=down, 2=left, 3=right (absolute).
Self-reversal turns are ignored (snake keeps last direction).
"""

import numpy as np

GRID = 64
CELL = 1
SIZE = 64  # output frame is always 64x64

COL_BG = (15, 15, 25)
COL_BODY = (90, 200, 110)
COL_HEAD = (240, 240, 80)
COL_FOOD = (230, 80, 80)

DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


class Snake:
    def __init__(self, seed=None, grid_cells=64):
        self.rng = np.random.default_rng(seed)
        self.grid = grid_cells
        self.cell = SIZE // grid_cells  # px per cell
        self.reset()

    def reset(self):
        cy, cx = self.grid // 2, self.grid // 2
        # Initial snake length: shorter for smaller grids
        init_len = min(5, max(2, self.grid // 4))
        self.body = [(cy, (cx - i) % self.grid) for i in range(init_len)]
        self.dir = 3
        self._place_food()
        self.done = False
        self.steps = 0
        return self.render()

    def _place_food(self):
        occupied = set(self.body)
        free = [(r, c) for r in range(self.grid) for c in range(self.grid) if (r, c) not in occupied]
        self.food = free[self.rng.integers(len(free))] if free else None

    def step(self, action):
        if self.done:
            return self.render(), True
        if action != OPPOSITE[self.dir]:
            self.dir = action
        dy, dx = DIRS[self.dir]
        head = self.body[0]
        new_head = ((head[0] + dy) % self.grid, (head[1] + dx) % self.grid)
        if new_head in self.body[:-1]:
            self.done = True
            return self.render(), True
        ate = new_head == self.food
        self.body.insert(0, new_head)
        if ate:
            self._place_food()
        else:
            self.body.pop()
        self.steps += 1
        return self.render(), False

    def render(self):
        img = np.full((SIZE, SIZE, 3), COL_BG, dtype=np.uint8)
        c = self.cell
        if self.food is not None:
            r, x = self.food
            img[r * c:(r + 1) * c, x * c:(x + 1) * c] = COL_FOOD
        for i, (r, x) in enumerate(self.body):
            color = COL_HEAD if i == 0 else COL_BODY
            img[r * c:(r + 1) * c, x * c:(x + 1) * c] = color
        return img


def heuristic_action(env):
    """Slightly-smart policy: prefer food direction, avoid immediate death."""
    head = env.body[0]
    food = env.food
    candidates = []
    if food is not None:
        if food[0] < head[0]:
            candidates.append(0)
        elif food[0] > head[0]:
            candidates.append(1)
        if food[1] < head[1]:
            candidates.append(2)
        elif food[1] > head[1]:
            candidates.append(3)
    candidates.extend([0, 1, 2, 3])

    body_set = set(env.body[:-1])
    for a in candidates:
        if a == OPPOSITE[env.dir]:
            continue
        dy, dx = DIRS[a]
        nh = ((head[0] + dy) % env.grid, (head[1] + dx) % env.grid)
        if nh not in body_set:
            return a
    return env.dir


def generate_episode(seed, max_len=200, p_random=0.15, grid_cells=64):
    env = Snake(seed=seed, grid_cells=grid_cells)
    rng = np.random.default_rng(seed + 9_000_000)
    frames, actions = [env.render()], []
    while len(actions) < max_len:
        a = rng.integers(4) if rng.random() < p_random else heuristic_action(env)
        frame, done = env.step(int(a))
        actions.append(int(a))
        frames.append(frame)
        if done:
            break
    return np.stack(frames), np.array(actions, dtype=np.int64)


def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _sinusoidal(x, n_freqs=8):
    """Fourier features: [sin(2^k*pi*x), cos(2^k*pi*x)] for k=0..n_freqs-1."""
    out = []
    for k in range(n_freqs):
        f = 2 ** k * np.pi
        out.extend([np.sin(f * x), np.cos(f * x)])
    return np.array(out, dtype=np.float32)


def _state_sinusoidal(env, head_y, head_x, food_y, food_x, n_freqs=8):
    feats = []
    n = 2 * n_freqs
    for v in (head_y / GRID, head_x / GRID, food_y / GRID, food_x / GRID):
        feats.append(_sinusoidal(v, n_freqs=n_freqs))
    feats.append(np.array([len(env.body) / 50.0], dtype=np.float32))
    for i in range(30):
        if i + 1 < len(env.body):
            y, x = env.body[i + 1]
            feats.append(_sinusoidal(y / GRID, n_freqs=n_freqs))
            feats.append(_sinusoidal(x / GRID, n_freqs=n_freqs))
            feats.append(np.array([1.0], dtype=np.float32))
        else:
            feats.append(np.zeros(n, dtype=np.float32))
            feats.append(np.zeros(n, dtype=np.float32))
            feats.append(np.array([0.0], dtype=np.float32))
    d = np.zeros(4, dtype=np.float32)
    d[env.dir] = 1.0
    feats.append(d)
    return np.concatenate(feats)


def state_features_v2(env, encoding="baseline"):
    """Multiple state-encoding variants for the precision ablation.
    Returns a 1-D float vector of variable length OR a 4x64x64 spatial mask
    (encoding='spatial').
    """
    if encoding == "baseline":
        return state_features(env)

    head_y, head_x = env.body[0]
    food_y, food_x = env.food if env.food is not None else (-1, -1)

    if encoding == "onehot":
        feats = []
        feats.append(_onehot(head_y, GRID))
        feats.append(_onehot(head_x, GRID))
        feats.append(_onehot(food_y, GRID))
        feats.append(_onehot(food_x, GRID))
        feats.append(np.array([len(env.body) / 50.0], dtype=np.float32))
        for i in range(30):
            if i + 1 < len(env.body):
                y, x = env.body[i + 1]
                feats.append(_onehot(y, GRID))
                feats.append(_onehot(x, GRID))
                feats.append(np.array([1.0], dtype=np.float32))
            else:
                feats.append(np.zeros(GRID, dtype=np.float32))
                feats.append(np.zeros(GRID, dtype=np.float32))
                feats.append(np.array([0.0], dtype=np.float32))
        d = np.zeros(4, dtype=np.float32)
        d[env.dir] = 1.0
        feats.append(d)
        return np.concatenate(feats)

    if encoding == "sinusoidal":
        return _state_sinusoidal(env, head_y, head_x, food_y, food_x, n_freqs=8)
    if encoding == "sinusoidal-K32":
        return _state_sinusoidal(env, head_y, head_x, food_y, food_x, n_freqs=32)

    if encoding == "spatial":
        # 4-channel binary mask: [head, body, food, empty]
        m = np.zeros((4, GRID, GRID), dtype=np.float32)
        if 0 <= head_y < GRID and 0 <= head_x < GRID:
            m[0, head_y, head_x] = 1.0
        for (y, x) in env.body[1:]:
            if 0 <= y < GRID and 0 <= x < GRID:
                m[1, y, x] = 1.0
        if env.food is not None and 0 <= food_y < GRID and 0 <= food_x < GRID:
            m[2, food_y, food_x] = 1.0
        m[3] = 1.0 - (m[0] + m[1] + m[2])
        return m

    raise ValueError(f"unknown state encoding: {encoding}")


def state_features(env):
    """Hand-crafted ground-truth state features for the oracle encoder.
    Returns a 99-d float vector capturing head, food, body, and direction.

    Layout: [head_y/G, head_x/G, food_y/G, food_x/G, len/50,
             then 30x (body_y/G, body_x/G, valid_flag),
             then direction one-hot (4)]   ->  2+2+1+90+4 = 99
    """
    feats = []
    head_y, head_x = env.body[0]
    feats.extend([head_y / GRID, head_x / GRID])
    if env.food is not None:
        feats.extend([env.food[0] / GRID, env.food[1] / GRID])
    else:
        feats.extend([-1.0, -1.0])
    feats.append(len(env.body) / 50.0)
    for i in range(30):
        if i + 1 < len(env.body):
            y, x = env.body[i + 1]
            feats.extend([y / GRID, x / GRID, 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0])
    dirs = [0.0, 0.0, 0.0, 0.0]
    dirs[env.dir] = 1.0
    feats.extend(dirs)
    return np.array(feats, dtype=np.float32)


def generate_episode_with_state(seed, max_len=200, p_random=0.15):
    """Same as generate_episode but also returns per-frame state features
    aligned with the frames axis."""
    env = Snake(seed=seed)
    rng = np.random.default_rng(seed + 9_000_000)
    frames = [env.render()]
    states = [state_features(env)]
    actions = []
    while len(actions) < max_len:
        a = rng.integers(4) if rng.random() < p_random else heuristic_action(env)
        frame, done = env.step(int(a))
        actions.append(int(a))
        frames.append(frame)
        states.append(state_features(env))
        if done:
            break
    return np.stack(frames), np.array(actions, dtype=np.int64), np.stack(states)


def generate_dataset(num_episodes, seed=0, grid_cells=64):
    """Returns (frames_list, actions_list) of variable lengths."""
    frames_all, actions_all = [], []
    for i in range(num_episodes):
        f, a = generate_episode(seed + i, grid_cells=grid_cells)
        frames_all.append(f)
        actions_all.append(a)
    return frames_all, actions_all


def generate_oracle_dataset(num_episodes, seed=0, encoding="baseline", return_actions=False):
    """Returns (frames_list, states_list[, actions_list]).
    states_list[i] shape: (T, D) for baseline/onehot/sinusoidal, (T, 4, 64, 64) for spatial.
    actions_list[i] shape: (T-1,) — action that took states[i][t] -> states[i][t+1]."""
    rng = np.random.default_rng(seed + 9_000_000)
    frames_all, states_all, actions_all = [], [], []
    for i in range(num_episodes):
        env = Snake(seed=seed + i)
        frames = [env.render()]
        states = [state_features_v2(env, encoding=encoding)]
        actions = []
        while len(actions) < 200:
            a = rng.integers(4) if rng.random() < 0.15 else heuristic_action(env)
            frame, done = env.step(int(a))
            actions.append(int(a))
            frames.append(frame)
            states.append(state_features_v2(env, encoding=encoding))
            if done:
                break
        frames_all.append(np.stack(frames))
        states_all.append(np.stack(states))
        actions_all.append(np.array(actions, dtype=np.int64))
    if return_actions:
        return frames_all, states_all, actions_all
    return frames_all, states_all


if __name__ == "__main__":
    fs, acs = generate_episode(0)
    print("frames:", fs.shape, "actions:", acs.shape, "lengths consistent:", len(fs) == len(acs) + 1)
