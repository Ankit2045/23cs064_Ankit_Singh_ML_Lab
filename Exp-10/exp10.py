import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.set_printoptions(precision=3, suppress=True)

ROWS, COLS = 3, 4
WALL = (1,1)
GOAL = (0,3)
PIT  = (1,3)
ACTIONS = ['up','right','down','left']
DELTA = {'up':(-1,0), 'right':(0,1), 'down':(1,0), 'left':(0,-1)}

P_INT = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1
GAMMA = 0.99

def all_states():
    s = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) != WALL:
                s.append((r,c))
    return s

STATES = all_states()

def is_terminal(s):
    return s in [GOAL, PIT]

def inside(r,c):
    return 0 <= r < ROWS and 0 <= c < COLS

LEFT = {'up':'left','left':'down','down':'right','right':'up'}
RIGHT = {'up':'right','right':'down','down':'left','left':'up'}

def attempt_move(state, action):
    if is_terminal(state):
        return state
    dr,dc = DELTA[action]
    nr, nc = state[0]+dr, state[1]+dc
    if not inside(nr,nc) or (nr,nc)==WALL:
        return state
    return (nr,nc)

def next_states(state, action):
    if is_terminal(state):
        return [(1.0, state)]
    a1 = attempt_move(state, action)
    a2 = attempt_move(state, LEFT[action])
    a3 = attempt_move(state, RIGHT[action])
    prob = {}
    for p,s in [(P_INT,a1),(P_LEFT,a2),(P_RIGHT,a3)]:
        prob[s] = prob.get(s,0)+p
    return [(prob[s],s) for s in prob]

def value_iteration(rewards, gamma=GAMMA, theta=1e-4, max_iters=10000):
    V = {s:0.0 for s in STATES}
    it = 0
    while True:
        it += 1
        delta = 0
        prev = V.copy()
        for s in STATES:
            if is_terminal(s):
                V[s] = rewards[s]
                continue
            qvals = []
            for a in ACTIONS:
                q = 0
                for p,ns in next_states(s,a):
                    r = rewards[ns]
                    q += p*(r + gamma*prev[ns])
                qvals.append(q)
            best = max(qvals)
            delta = max(delta, abs(best - prev[s]))
            V[s] = best
        if delta < theta or it>=max_iters:
            break
    return V, it

def extract_policy(V, rewards, gamma=GAMMA):
    policy = {}
    for s in STATES:
        if is_terminal(s):
            policy[s] = None
            continue
        best_a = None
        best_q = -1e9
        for a in ACTIONS:
            q = 0
            for p,ns in next_states(s,a):
                r = rewards[ns]
                q += p*(r + gamma*V[ns])
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy

ARROW = {'up':'↑','down':'↓','left':'←','right':'→',None:'·'}

def plot_values(V, title):
    grid = np.zeros((ROWS,COLS))
    mask = np.zeros((ROWS,COLS),dtype=bool)
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c)==WALL:
                grid[r,c] = np.nan
                mask[r,c] = True
            else:
                grid[r,c] = V[(r,c)]
    plt.figure(figsize=(6,4))
    ax = sns.heatmap(grid, annot=True, fmt=".3f", mask=mask, cmap="viridis")
    ax.invert_yaxis()
    plt.title(title)
    plt.show()

def plot_policy(policy, title):
    arr = np.empty((ROWS,COLS),dtype=object)
    for r in range(ROWS):
        for c in range(COLS):
            s = (r,c)
            if s==WALL: arr[r,c] = 'W'
            elif s==GOAL: arr[r,c] = 'G'
            elif s==PIT: arr[r,c] = 'P'
            else: arr[r,c] = ARROW[policy[s]]
    print(title)
    for r in range(ROWS):
        print(" ".join(f"{arr[r,c]:>3}" for c in range(COLS)))
    print()

    plt.figure(figsize=(6,4))
    bg = np.zeros((ROWS,COLS))
    bg[WALL] = np.nan
    ax = sns.heatmap(bg, annot=arr, fmt='', cbar=False, linewidths=.5, linecolor='gray')
    ax.invert_yaxis()
    plt.title(title)
    plt.show()

def run(living_penalty):
    rewards = {}
    for s in STATES:
        if s==GOAL: rewards[s]=1
        elif s==PIT: rewards[s]=-1
        else: rewards[s]=living_penalty
    V,it = value_iteration(rewards)
    P = extract_policy(V,rewards)
    print(f"Living penalty = {living_penalty}, iterations = {it}")
    plot_values(V, f"Value Function (r={living_penalty})")
    plot_policy(P, f"Policy (r={living_penalty})")
    return V,P

V1,P1 = run(-0.04)   # default
V2,P2 = run(0.0)     # no penalty
V3,P3 = run(-0.5)    # strong penalty
