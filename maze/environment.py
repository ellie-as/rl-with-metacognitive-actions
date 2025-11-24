import gymnasium as gym
import numpy as np
import numpy
from gymnasium import spaces
import random
import torch
from collections import deque
from utils import one_hot_encode
from config import Config


class MazeEnv(gym.Env):
    """
    Real environment with BFS-based barrier generation and
    an observation that encodes (agent position, goal position)
    via one-hot of size grid_size^2 each, concatenated => shape (2 * grid_size^2,).
    """
    def __init__(self, grid_size=3):
        super().__init__()
        self.grid_size = grid_size
        # Observations: (2 * grid_size^2,)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(2 * self.grid_size ** 2,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.maze = None         # 2D array: 0=free, 1=barrier
        self.state = None        # agent's current cell (integer 0..g*g-1)
        self.goal_state = None   # goal cell (integer distinct from self.state)
        self.max_steps = 30
        self.current_steps = 0

        self.last_removed_square = None
        self.last_changed_cells = []
        # Avoid oscillating between the same add/remove pairs
        self._recent_change_pairs = deque(maxlen=3)
        # Track which cells were touched by incremental updates since the last full reset
        self._cells_changed_since_full_update = set()
        self.allowed_pairs = None  # Optional list of (start_idx, goal_idx) used when resetting

        # Pre-initialize some random barriers with the default (non-incremental) behavior.
        self.reset_barriers()
        self._pending_terminal = False

    def reset(self, seed=None, options=None):
        """
        Reset the episode by placing the agent in a free cell
        and a distinct free goal cell. Then produce an observation.
        """
        super().reset(seed=seed)
        if self.maze is None:
            self.reset_barriers()  # if still none, set it

        self.current_steps = 0
        pair = None
        if options and "start_idx" in options and "goal_idx" in options:
            pair = (int(options["start_idx"]), int(options["goal_idx"]))
        elif self.allowed_pairs:
            pair = random.choice(self.allowed_pairs)
        self._reset_goal(pair)
        return self._get_observation(), {}

    def _reset_goal(self, forced_pair=None):
        """
        Randomly choose a free start cell for agent, and
        a different free cell for the goal.
        """
        free_cells = [(r, c)
                      for r in range(self.grid_size)
                      for c in range(self.grid_size)
                      if self.maze[r, c] == 0]  # 0 means free

        if not free_cells:
            # Edge case: if the entire maze is blocked (shouldn't happen with BFS approach),
            # fallback to a forced no-barrier reset:
            print("Warning: Maze has no free cells; regenerating barriers.")
            self.reset_barriers()
            free_cells = [(r, c)
                          for r in range(self.grid_size)
                          for c in range(self.grid_size)
                          if self.maze[r, c] == 0]

        chosen = None
        if forced_pair is not None:
            start_idx, goal_idx = forced_pair
            start_r, start_c = divmod(start_idx, self.grid_size)
            goal_r, goal_c = divmod(goal_idx, self.grid_size)
            allow_blocked_goals = bool(getattr(Config, "INCLUDE_IMPOSSIBLE_GOALS", False))
            start_in_bounds = 0 <= start_r < self.grid_size and 0 <= start_c < self.grid_size
            goal_in_bounds = 0 <= goal_r < self.grid_size and 0 <= goal_c < self.grid_size
            start_free = start_in_bounds and self.maze[start_r, start_c] == 0
            goal_free = goal_in_bounds and self.maze[goal_r, goal_c] == 0
            goal_allowed = goal_free or (allow_blocked_goals and goal_in_bounds)
            if start_free and goal_allowed:
                chosen = ((start_r, start_c), (goal_r, goal_c))
            else:
                print("Forced start/goal invalid or blocked; falling back to random selection.")

        if chosen is None:
            start_r, start_c = random.choice(free_cells)
            goal_r, goal_c = random.choice(free_cells)
        else:
            (start_r, start_c), (goal_r, goal_c) = chosen

        self.state = start_r * self.grid_size + start_c
        self.goal_state = goal_r * self.grid_size + goal_c
        self._pending_terminal = (self.state == self.goal_state)

    def reset_barriers(self, incremental=False):
        """
        Randomly place barriers in the grid, ensuring BFS connectivity among free cells.
        If incremental is False (default), then the maze is reset completely (i.e. all cells
        are set to free and then barriers are added up to ~50% of cells if possible).
        If incremental is True, then a single barrier is added to one random free cell,
        only if connectivity is maintained.
        """
        g = self.grid_size
        prev_maze = None if self.maze is None else self.maze.copy()
        self.last_changed_cells = []

        def is_connected(maze):
            """
            BFS over free cells in the provided maze array.
            Return True if all free cells are reachable.
            """
            free_positions = [(r, c) for r in range(g) for c in range(g) if maze[r, c] == 0]
            if not free_positions:
                return False
            visited = set()
            queue = [free_positions[0]]
            visited.add(free_positions[0])
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while queue:
                rr, cc = queue.pop(0)
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < g and 0 <= nc < g:
                        if maze[nr, nc] == 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            return len(visited) == len(free_positions)

        if incremental:
            # Ensure maze exists; if not, initialize to all free cells.
            if self.maze is None:
                self.maze = np.zeros((g, g), dtype=int)
            free_cells = [(r, c) for r in range(g) for c in range(g) if self.maze[r, c] == 0]
            barrier_cells = [(r, c) for r in range(g) for c in range(g) if self.maze[r, c] == 1]
            if not free_cells:
                print("No free cells available to add a barrier.")
                return
            if not barrier_cells:
                print("No barrier cells available to remove; cannot perform incremental update.")
                return

            changed_since_full = self._cells_changed_since_full_update
            eligible_free_cells = [cell for cell in free_cells if cell not in changed_since_full]
            eligible_barrier_cells = [cell for cell in barrier_cells if cell not in changed_since_full]

            if not eligible_free_cells:
                print("No eligible free cells remain for incremental update; all were modified since the last full reset.")
                return
            if not eligible_barrier_cells:
                print("No eligible barrier cells remain for incremental update; all were modified since the last full reset.")
                return

            max_attempts = 100
            success = False
            attempts = 0
            def _free_degree(rr, cc, maze_arr):
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                deg = 0
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < g and 0 <= nc < g and maze_arr[nr, nc] == 0:
                        deg += 1
                return deg

            def _no_new_dead_ends(maze_arr, added_barrier, newly_free):
                # Newly freed cell should not be a dead end
                if _free_degree(newly_free[0], newly_free[1], maze_arr) < 2:
                    return False
                # Adding a barrier should not create dead-ends in its neighbors
                rr, cc = added_barrier
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < g and 0 <= nc < g and maze_arr[nr, nc] == 0:
                        if _free_degree(nr, nc, maze_arr) < 2:
                            return False
                return True

            # Helper: iterate free neighbors
            def _neighbors(cell, maze_arr):
                r, c = cell
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < g and 0 <= nc < g and maze_arr[nr, nc] == 0:
                        yield (nr, nc)

            strategy = str(getattr(Config, "INCREMENTAL_UPDATE_STRATEGY", "centrality")).lower()
            use_centrality = strategy == "centrality"

            if use_centrality:
                # Compute node betweenness centrality on current free graph
                def _betweenness_centrality_nodes(maze_arr):
                    from collections import deque
                    nodes = [(r, c) for r in range(g) for c in range(g) if maze_arr[r, c] == 0]
                    Cb = {node: 0.0 for node in nodes}
                    for s in nodes:
                        S = []
                        P = {v: [] for v in nodes}
                        sigma = {v: 0.0 for v in nodes}
                        dist = {v: -1 for v in nodes}
                        sigma[s] = 1.0
                        dist[s] = 0
                        Q = deque([s])
                        while Q:
                            v = Q.popleft()
                            S.append(v)
                            for w in _neighbors(v, maze_arr):
                                if dist[w] < 0:
                                    Q.append(w)
                                    dist[w] = dist[v] + 1
                                if dist[w] == dist[v] + 1:
                                    sigma[w] += sigma[v]
                                    P[w].append(v)
                        delta = {v: 0.0 for v in nodes}
                        while S:
                            w = S.pop()
                            for v in P[w]:
                                if sigma[w] > 0:
                                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                            if w != s:
                                Cb[w] += delta[w]
                    # undirected graph adjustment
                    for k in Cb:
                        Cb[k] *= 0.5
                    return Cb

                # Rank candidates by centrality
                Cb_now = _betweenness_centrality_nodes(self.maze)
                free_sorted = sorted(eligible_free_cells, key=lambda x: Cb_now.get(x, 0.0), reverse=True)

                # For barriers, evaluate centrality if freed
                barrier_scores = []
                for cell in eligible_barrier_cells:
                    r, c = cell
                    self.maze[r, c] = 0
                    score = _betweenness_centrality_nodes(self.maze).get(cell, 0.0)
                    ok_dead = _free_degree(r, c, self.maze) >= 2
                    self.maze[r, c] = 1
                    if ok_dead:
                        barrier_scores.append((cell, score))
                barrier_sorted = [cell for cell, _ in sorted(barrier_scores, key=lambda x: x[1], reverse=True)]

            def _is_tabu(a_cell, b_cell):
                # Avoid repeating the same pair or its inverse
                return (a_cell, b_cell) in self._recent_change_pairs or (b_cell, a_cell) in self._recent_change_pairs

            if use_centrality:
                # Try top-k combinations
                top_k = 10
                for a_cell in free_sorted[:top_k]:
                    for b_cell in barrier_sorted[:top_k] if barrier_sorted else []:
                        if a_cell == b_cell:
                            continue
                        if _is_tabu(a_cell, b_cell):
                            continue
                        orig_val_add = self.maze[a_cell[0], a_cell[1]]
                        orig_val_remove = self.maze[b_cell[0], b_cell[1]]
                        self.maze[a_cell[0], a_cell[1]] = 1
                        self.maze[b_cell[0], b_cell[1]] = 0
                        if is_connected(self.maze) and _no_new_dead_ends(self.maze, a_cell, b_cell):
                            success = True
                            self.last_removed_square = b_cell
                            print("Incremental (centrality) update: barrier added at", a_cell, "and removed at", b_cell)
                            self._recent_change_pairs.append((a_cell, b_cell))
                            self._cells_changed_since_full_update.update({a_cell, b_cell})
                            break
                        # revert and continue
                        self.maze[a_cell[0], a_cell[1]] = orig_val_add
                        self.maze[b_cell[0], b_cell[1]] = orig_val_remove
                    if success:
                        break

            # Fallback to random attempts if no centrality pair found (or centrality disabled)
            while not success and attempts < max_attempts:
                attempts += 1
                candidate_add = random.choice(eligible_free_cells)
                candidate_remove = random.choice(eligible_barrier_cells)
                if candidate_add == candidate_remove:
                    continue
                if _is_tabu(candidate_add, candidate_remove):
                    continue
                orig_val_add = self.maze[candidate_add[0], candidate_add[1]]
                orig_val_remove = self.maze[candidate_remove[0], candidate_remove[1]]
                self.maze[candidate_add[0], candidate_add[1]] = 1
                self.maze[candidate_remove[0], candidate_remove[1]] = 0
                if is_connected(self.maze) and _no_new_dead_ends(self.maze, candidate_add, candidate_remove):
                    success = True
                    self.last_removed_square = candidate_remove
                    if use_centrality:
                        print("Incremental update succeeded: barrier added at", candidate_add, "and barrier removed at", candidate_remove)
                    else:
                        print("Incremental (random) update: barrier added at", candidate_add, "and barrier removed at", candidate_remove)
                    self._recent_change_pairs.append((candidate_add, candidate_remove))
                    self._cells_changed_since_full_update.update({candidate_add, candidate_remove})
                else:
                    self.maze[candidate_add[0], candidate_add[1]] = orig_val_add
                    self.maze[candidate_remove[0], candidate_remove[1]] = orig_val_remove
            if not success:
                mode = "centrality" if use_centrality else "random"
                print(f"Incremental ({mode}) update failed after {max_attempts} attempts.")
                self.last_removed_square = None

        else:
            # Full reset: initialize maze to all free cells.
            self.maze = np.zeros((g, g), dtype=int)
            self._cells_changed_since_full_update.clear()
            self._recent_change_pairs.clear()
            total_cells = g * g
            max_barriers = int(total_cells * 0.4)  # up to ~50% barriers
            barriers_added = 0
            while barriers_added < max_barriers:
                row = random.randint(0, g - 1)
                col = random.randint(0, g - 1)
                if self.maze[row, col] == 0:  # candidate for barrier
                    # Place barrier temporarily.
                    self.maze[row, col] = 1
                    if is_connected(self.maze):
                        barriers_added += 1
                    else:
                        # Revert if it breaks connectivity.
                        self.maze[row, col] = 0
            self.last_removed_square = None

            # Optional: reopen a configurable fraction of remaining barrier cells to create multiple paths
            if getattr(Config, "MAZE_MODE", "default") == "extra_links":
                barrier_cells = [(r, c) for r in range(g) for c in range(g) if self.maze[r, c] == 1]
                if barrier_cells:
                    random.shuffle(barrier_cells)
                    fraction = float(getattr(Config, "EXTRA_LINKS_REOPEN_FRACTION", 0.5))
                    fraction = min(max(fraction, 0.0), 1.0)
                    reopen_n = max(1, int(round(len(barrier_cells) * fraction)))
                    reopened = 0
                    for r, c in barrier_cells:
                        # Temporarily reopen and keep if connectivity among free cells remains
                        self.maze[r, c] = 0
                        if is_connected(self.maze):
                            reopened += 1
                            if reopened >= reopen_n:
                                break
                        else:
                            # revert if reopening disconnects free space (shouldn't normally)
                            self.maze[r, c] = 1

        if prev_maze is not None and self.maze is not None:
            diff_indices = np.argwhere(self.maze != prev_maze)
            self.last_changed_cells = [tuple(idx) for idx in diff_indices]
        else:
            self.last_changed_cells = []

    def step(self, action):
        if self._pending_terminal:
            self._pending_terminal = False
            info = {"maze": self.maze.copy()}
            return self._get_observation(), 1.0, True, False, info

        action = int(action)
        """
        Convert self.state to (row, col). Attempt to move.
        If the new cell is free, update self.state.
        Reward = 1 if agent reaches goal, else -0.05.
        Episode ends if agent is at goal or max_steps are exceeded.
        """
        row, col = divmod(self.state, self.grid_size)
        # 0: up, 1: down, 2: left, 3: right
        if action == 0 and row > 0 and self.maze[row - 1, col] == 0:
            row -= 1
        elif action == 1 and row < self.grid_size - 1 and self.maze[row + 1, col] == 0:
            row += 1
        elif action == 2 and col > 0 and self.maze[row, col - 1] == 0:
            col -= 1
        elif action == 3 and col < self.grid_size - 1 and self.maze[row, col + 1] == 0:
            col += 1

        new_state = row * self.grid_size + col
        reward = 1.0 if new_state == self.goal_state else -0.01
        done = (new_state == self.goal_state)

        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            done = True

        self.state = new_state

        # Include the maze barriers in info so that the visualizer can show them.
        info = {"maze": self.maze.copy()}

        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        """
        Construct an observation of shape (2 * grid_size^2,):
          - first grid_size^2 is a one-hot of agent position,
          - second grid_size^2 is a one-hot of goal position.
        """
        agent_oh = one_hot_encode(self.state, self.grid_size)
        goal_oh = one_hot_encode(self.goal_state, self.grid_size)
        return np.concatenate([agent_oh, goal_oh], axis=0)

    def _introduce_alternate_routes(self):
        if self.maze is None:
            return
        g = self.grid_size
        candidates = []
        for r in range(g):
            for c in range(g):
                if self.maze[r, c] != 1:
                    continue
                free_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < g and 0 <= nc < g and self.maze[nr, nc] == 0:
                        free_neighbors += 1
                if free_neighbors >= 2:
                    candidates.append((r, c))

        if not candidates:
            return

        random.shuffle(candidates)
        reopen_target = max(1, min(len(candidates), g // 3))
        for r, c in candidates[:reopen_target]:
            self.maze[r, c] = 0


class DreamEnv(gym.Env):
    """
    'Imagined' environment. Also uses a one-hot approach for (agent position, goal position)
    resulting in an observation of size 2*g*g. Transitions come from self.world_model(...).
    
    Optionally, an allowed_pairs list can be provided such that when reset is called without explicit
    options, a (start, goal) pair is randomly selected from that list.
    """
    def __init__(self, world_model, obs_dim, action_dim, barriers=None, allowed_pairs=None):
        super().__init__()
        self.world_model = world_model
        self.obs_dim = obs_dim     # Expect 2*g*g
        self.action_dim = action_dim
        # barriers may be a 2D maze array (preferred) or a list/tuple of blocked cells
        if barriers is None:
            self.barriers = []
        else:
            self.barriers = barriers
        self.max_steps = 30
        self.current_steps = 0

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.state_idx = None      # agent's position index
        self.goal_idx = None       # goal's index
        self.grid_size = int(np.sqrt(obs_dim // 2))
        self.allowed_pairs = list(allowed_pairs) if allowed_pairs is not None else []
        self._allowed_queue = []
        self._refresh_allowed_queue()
        self._pending_terminal = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps = 0

        # Priority: use options if provided.
        if options and 'start_idx' in options and 'goal_idx' in options:
            self.state_idx = options['start_idx']
            self.goal_idx = options['goal_idx']
        # Next, if allowed_pairs is available, select one at random.
        elif self.allowed_pairs:
            if not self._allowed_queue:
                self._refresh_allowed_queue()
            start, goal = self._allowed_queue.pop()
            self.state_idx = start
            self.goal_idx = goal
        else:
            # Fallback: sample only from free cells if known; else uniform over all cells
            free_cells = self._free_cells()
            if free_cells:
                start = random.choice(free_cells)
                goal = start
                while goal == start:
                    goal = random.choice(free_cells)
                self.state_idx = start
                self.goal_idx = goal
            else:
                num_cells = self.grid_size * self.grid_size
                start = random.randint(0, num_cells - 1)
                goal = start
                while goal == start:
                    goal = random.randint(0, num_cells - 1)
                self.state_idx = start
                self.goal_idx = goal

        self._pending_terminal = (self.state_idx == self.goal_idx)
        return self._get_observation(), {}

    def step(self, action):
        if self._pending_terminal:
            self._pending_terminal = False
            return self._get_observation(), 1.0, True, False, {}

        action = int(action)
        if self.world_model is None:
            done = (self.state_idx == self.goal_idx)
            reward = 1.0 if done else -0.01
            self.current_steps += 1
            if self.current_steps >= self.max_steps:
                done = True
            return self._get_observation(), reward, done, False, {}

        agent_oh = one_hot_encode(self.state_idx, self.grid_size)
        goal_oh = one_hot_encode(self.goal_idx, self.grid_size)
        input_vec = np.concatenate([agent_oh, goal_oh], axis=0)

        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        action_one_hot = torch.nn.functional.one_hot(
            action_tensor, num_classes=self.action_dim
        ).float().squeeze(1)

        with torch.no_grad():
            output = self.world_model(input_tensor, action_one_hot)
            full_logits = output[0].numpy()
            # Interpret the first g*g outputs as the agent's next position.
            next_idx = np.argmax(full_logits[: (self.grid_size * self.grid_size)])

        self.state_idx = next_idx
        done = (next_idx == self.goal_idx)
        reward = 1.0 if done else -0.01

        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        agent_oh = one_hot_encode(self.state_idx, self.grid_size)
        goal_oh = one_hot_encode(self.goal_idx, self.grid_size)
        return np.concatenate([agent_oh, goal_oh], axis=0)

    def _refresh_allowed_queue(self):
        """Shuffle allowed pairs so each is used once before repeating."""
        self._allowed_queue = list(self.allowed_pairs)
        if self._allowed_queue:
            random.shuffle(self._allowed_queue)

    def _free_cells(self):
        """
        Return a list of free cell indices if known. Prefer explicit maze array (barriers as 2D array).
        Fallback to world_model cache keys when using cache/debug; else return None.
        """
        # If barriers provided as a 2D numpy array (maze): 0 = free, 1 = barrier
        try:
            import numpy as _np
            if isinstance(self.barriers, _np.ndarray) and self.barriers.ndim == 2:
                g = self.barriers.shape[0]
                cells = [r * g + c for r in range(g) for c in range(g) if self.barriers[r, c] == 0]
                return cells
        except Exception:
            pass

        # If cache-based model, derive reachable states from cached keys
        wm = getattr(self, "world_model", None)
        cache = getattr(wm, "cache", None)
        if isinstance(cache, dict) and len(cache) > 0:
            states = sorted({int(k[0]) for k in cache.keys()})
            return states
        return None
