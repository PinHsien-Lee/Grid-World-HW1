from flask import Flask, render_template, request, jsonify
import random
import math

app = Flask(__name__)

# ── Security constants ────────────────────────────────────────
N_MIN            = 5
N_MAX            = 9          # grid dimension hard cap
NOISE_MIN        = 0.0
NOISE_MAX        = 1.0
GAMMA_MIN        = 0.1
GAMMA_MAX        = 1.0
STEP_COST_MIN    = -1.0
STEP_COST_MAX    = 0.0
MAX_OBSTACLES    = N_MAX * N_MAX  # absolute upper bound


def validate_evaluate_input(data):
    """Validate and sanitize all fields from the /evaluate request.
    Returns (clean_dict, error_message). On success error_message is None."""
    errors = []

    # ── grid size ─────────────────────────────────────────────
    try:
        n = int(data['n'])
    except (KeyError, ValueError, TypeError):
        return None, "'n' must be an integer."
    if not (N_MIN <= n <= N_MAX):
        return None, f"'n' must be between {N_MIN} and {N_MAX}. Got {n}."

    # ── start cell ────────────────────────────────────────────
    try:
        start = (int(data['start']['row']), int(data['start']['col']))
    except (KeyError, ValueError, TypeError):
        return None, "'start' must have integer 'row' and 'col'."
    if not (0 <= start[0] < n and 0 <= start[1] < n):
        return None, f"'start' {start} is out of grid bounds (0–{n-1})."

    # ── end cell ──────────────────────────────────────────────
    try:
        end = (int(data['end']['row']), int(data['end']['col']))
    except (KeyError, ValueError, TypeError):
        return None, "'end' must have integer 'row' and 'col'."
    if not (0 <= end[0] < n and 0 <= end[1] < n):
        return None, f"'end' {end} is out of grid bounds (0–{n-1})."
    if start == end:
        return None, "'start' and 'end' cannot be the same cell."

    # ── obstacles ─────────────────────────────────────────────
    raw_obs = data.get('obstacles', [])
    if not isinstance(raw_obs, list):
        return None, "'obstacles' must be a list."
    if len(raw_obs) > MAX_OBSTACLES:
        return None, f"Too many obstacles (max {MAX_OBSTACLES})."

    obstacles = set()
    for i, o in enumerate(raw_obs):
        try:
            r, c = int(o['row']), int(o['col'])
        except (KeyError, ValueError, TypeError):
            return None, f"Obstacle #{i} has invalid 'row'/'col'."
        if not (0 <= r < n and 0 <= c < n):
            return None, f"Obstacle #{i} {(r,c)} is out of grid bounds."
        if (r, c) == start:
            return None, f"Obstacle #{i} overlaps with start cell."
        if (r, c) == end:
            return None, f"Obstacle #{i} overlaps with end cell."
        obstacles.add((r, c))

    # ── continuous parameters ──────────────────────────────────
    try:
        noise = float(data.get('noise', 0.0))
        if math.isnan(noise) or math.isinf(noise):
            raise ValueError
        noise = max(NOISE_MIN, min(NOISE_MAX, noise))
    except (ValueError, TypeError):
        return None, "'noise' must be a finite float in [0, 1]."

    try:
        gamma = float(data.get('gamma', 0.9))
        if math.isnan(gamma) or math.isinf(gamma):
            raise ValueError
        gamma = max(GAMMA_MIN, min(GAMMA_MAX, gamma))
    except (ValueError, TypeError):
        return None, "'gamma' must be a finite float in [0.1, 1]."

    try:
        step_cost = float(data.get('step_cost', -0.04))
        if math.isnan(step_cost) or math.isinf(step_cost):
            raise ValueError
        step_cost = max(STEP_COST_MIN, min(STEP_COST_MAX, step_cost))
    except (ValueError, TypeError):
        return None, "'step_cost' must be a finite float in [-1, 0]."

    return {
        'n':         n,
        'start':     start,
        'end':       end,
        'obstacles': frozenset(obstacles),
        'noise':     noise,
        'gamma':     gamma,
        'step_cost': step_cost,
    }, None


def build_response(V, policy, n, start, end, obstacles):
    cells = {}
    for r in range(n):
        for c in range(n):
            key = f"{r}_{c}"
            if (r, c) in obstacles:
                cells[key] = {'type': 'obstacle', 'actions': [], 'value': ''}
            elif (r, c) == end:
                cells[key] = {'type': 'end', 'actions': [], 'value': '1.000'}
            else:
                cells[key] = {
                    'type': 'start' if (r, c) == start else 'normal',
                    'actions': policy[(r, c)],
                    'value': f"{V[(r, c)]:.3f}"
                }
    return cells

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON.'}), 400

    clean, err = validate_evaluate_input(data)
    if err:
        return jsonify({'error': err}), 400

    n         = clean['n']
    start     = clean['start']
    end       = clean['end']
    obstacles = clean['obstacles']
    noise     = clean['noise']
    gamma     = clean['gamma']
    step_cost = clean['step_cost']

    actions = ['up', 'down', 'left', 'right']

    def get_transitions(r, c, a):
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        ortho = {'up': ['left', 'right'], 'down': ['left', 'right'], 'left': ['up', 'down'], 'right': ['up', 'down']}
        
        transitions = []
        # Normal intended action
        dr, dc = moves[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles:
            transitions.append((1.0 - noise, nr, nc))
        else:
            transitions.append((1.0 - noise, r, c))
            
        # Slipping due to noise
        if noise > 0:
            for oa in ortho[a]:
                dr, dc = moves[oa]
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles:
                    transitions.append((noise / 2, nr, nc))
                else:
                    transitions.append((noise / 2, r, c))
        return transitions

    # ── 1. Random Deterministic Policy Evaluation ─────────────
    random_policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) not in obstacles:
                random_policy[(r, c)] = [random.choice(actions)]

    theta = 1e-6

    V_rand = {(r, c): 0.0 for r in range(n) for c in range(n) if (r, c) not in obstacles}
    for _ in range(10000):
        delta = 0.0
        new_V = {}
        for r in range(n):
            for c in range(n):
                if (r, c) in obstacles:
                    continue
                if (r, c) == end:
                    new_V[(r, c)] = 0.0
                    continue
                a = random_policy[(r, c)][0]
                val = 0.0
                for prob, nr, nc in get_transitions(r, c, a):
                    reward = 1.0 if (nr, nc) == end else step_cost
                    val += prob * (reward + gamma * V_rand.get((nr, nc), 0.0))
                new_V[(r, c)] = val
                delta = max(delta, abs(new_V[(r, c)] - V_rand[(r, c)]))
        V_rand = new_V
        if delta < theta:
            break

    rand_cells = build_response(V_rand, random_policy, n, start, end, obstacles)

    # ── 2. Value Iteration (Optimal Policy) ───────────────────
    V_opt = {(r, c): 0.0 for r in range(n) for c in range(n) if (r, c) not in obstacles}
    for _ in range(10000):
        delta = 0.0
        new_V = {}
        for r in range(n):
            for c in range(n):
                if (r, c) in obstacles:
                    continue
                if (r, c) == end:
                    new_V[(r, c)] = 0.0
                    continue
                max_val = float('-inf')
                for a in actions:
                    val = 0.0
                    for prob, nr, nc in get_transitions(r, c, a):
                        reward = 1.0 if (nr, nc) == end else step_cost
                        val += prob * (reward + gamma * V_opt.get((nr, nc), 0.0))
                    if val > max_val:
                        max_val = val
                new_V[(r, c)] = max_val
                delta = max(delta, abs(new_V[(r, c)] - V_opt[(r, c)]))
        V_opt = new_V
        if delta < theta:
            break

    # Extract deterministic optimal policy
    opt_policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) in obstacles or (r, c) == end:
                continue
            max_val = float('-inf')
            best_a = []
            for a in actions:
                val = 0.0
                for prob, nr, nc in get_transitions(r, c, a):
                    reward = 1.0 if (nr, nc) == end else step_cost
                    val += prob * (reward + gamma * V_opt.get((nr, nc), 0.0))
                # Add tiny epsilon for float comparison
                if abs(val - max_val) < 1e-6:
                    best_a.append(a)
                elif val > max_val:
                    max_val = val
                    best_a = [a]
            # Just take one optimal action to keep it clean, or all if you prefer
            opt_policy[(r, c)] = best_a

    opt_cells = build_response(V_opt, opt_policy, n, start, end, obstacles)

    # ── 3. Trace optimal path from start to end ────────────────
    opt_path = []
    visited_path = set()
    cur = start
    max_steps = n * n
    while cur != end and len(opt_path) <= max_steps:
        if cur in visited_path:
            break  # cycle detected
        visited_path.add(cur)
        opt_path.append({'row': cur[0], 'col': cur[1]})
        if cur not in opt_policy:
            break
        # Pick the action with the single best value to trace deterministically
        a = opt_policy[cur][0]
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        dr, dc = moves[a]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in obstacles:
            break
        cur = nxt
    # Include the end cell itself
    opt_path.append({'row': end[0], 'col': end[1]})

    return jsonify({
        'rand_cells': rand_cells,
        'opt_cells':  opt_cells,
        'opt_path':   opt_path
    })


if __name__ == '__main__':
    app.run(debug=True)