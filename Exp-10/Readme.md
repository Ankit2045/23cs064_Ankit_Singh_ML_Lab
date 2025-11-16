A Markov Decision Process (MDP) is a mathematical framework used to model decision-making in environments where outcomes are partly random and partly under an agent’s control. An MDP is defined by five components:

States (S)

Actions (A)

Transition probabilities (T)

Rewards (R)

Discount factor (γ)

In this experiment, a 3×4 GridWorld environment is modeled as an MDP. The agent can move in four directions, but movement is stochastic due to slipping probabilities. Certain cells act as terminal states (goal and pit), and one cell is a wall.

To solve the MDP, the Value Iteration algorithm is implemented from scratch. Value Iteration repeatedly applies the Bellman Optimality Equation to estimate the long-term value of every state:

𝑉
(
𝑠
)
=
max
⁡
𝑎
∑
𝑠
′
𝑇
(
𝑠
,
𝑎
,
𝑠
′
)
[
𝑅
(
𝑠
′
)
+
𝛾
𝑉
(
𝑠
′
)
]
V(s)=
a
max
	​

s
′
∑
	​

T(s,a,s
′
)[R(s
′
)+γV(s
′
)]

The process continues until convergence. After computing the final value function, an optimal policy is extracted by selecting, for each state, the action that yields the highest expected value.

The results are visualized using heatmaps for the value function and arrow maps for the optimal policy. By changing the “living penalty,” the experiment shows how reward shaping affects the agent’s behavior and path selection.

This experiment demonstrates the fundamentals of planning in reinforcement learning and how optimal decision-making emerges from iterative evaluation of state values.
