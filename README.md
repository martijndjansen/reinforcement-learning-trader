# notebook-reinforcement-learning

## Improvements
1. CNN or LSTM, transformer, or mambda over MLP
2. Implement Differential Sharpe Ratio
    $$R_t = DSR_t - \lambda \cdot (Transaction costs)_t$$

## Implementation
1. PPO + Dirichlet. PPO over SAC as SAC does not have Dirichlet 'support'


### Mathematical notation of agent

To ensure a realistic simulation, actions are taken before market movements are observed.

#### 1. Definitions
For any timestep $t \in \{1, 2, \dots, T\}$ and any asset $i \in \{1, 2, \dots, N\}$:

- $\mathbf{X}\_{t-1} \in \mathbb{R}^{H \times (N \cdot M)}$: The history of $M$ features for each of the $N$ stocks over $H$ timesteps.
- $\mathbf{u}\_{t-1} \in \mathbb{R}^{G}$: A vector of $G$ global features at the end of $t-1$.
- $\mathbf{w}\_{t} \in \mathbb{R}^{N+1}$: The portfolio weights held at the beginning of step $t$ (the drifted result of $\mathbf{a}_{t-1}$).
- $\mathbf{x}\_{t-1} = \left(\mathbf{X}\_{t-1}, \mathbf{u}\_{t-1}, \mathbf{w}\_{t}\right)$: The composite observation provided to the agent.
- $\mathbf{a}\_{t} \in \mathbb{R}^{N+1}$: The action vector (target weights) chosen by the agent:

$$\sum_{i=1}^{N+1} a_{t,i} = 1, \quad a_{t,i} \geq 0$$

- $v_t \in \mathbb{R}$: The total portfolio value at the end of timestep $t$.
- $f \in \mathbb{R}$: The transaction fee rate.

#### 2. Calculating new portfolio value
The price returns $\mathbf{r}_t \in \mathbb{R}^{N+1}$ represents the change in price of all stocks during timestep $t$:

$$\mathbf{r}_t = \frac{\mathbf{p}_t}{\mathbf{p}_{t-1} + \epsilon}$$

At each step $t$, the agent rebalances from the drifted weights $\mathbf{w}_t$ to the target $\mathbf{a}_t$, incurring a transaction cost $c_t$:

$$c_t = f \sum_{i=1}^{N+1} |\mathbf{a}_{t,i} - \mathbf{w}_{t,i}|$$

The portfolio's growth factor $G_t$ is the return realized by the new allocation:

$$G_t = \mathbf{a}_t^\top \mathbf{r}_t$$

The portfolio value $v_t$ is updated by accounting for capital lost to fees followed by market growth:

$$v_t = v_{t-1} \cdot (1 - c_t) \cdot G_t$$

#### 3. Weight Update (Drift)
After the market move $\mathbf{r}_t$, the weights for the next timestep $\mathbf{w}_{t+1}$ are calculated by the relative growth of each asset:

$$\mathbf{w}_{t+1} = \frac{\mathbf{a}_t \odot \mathbf{r}_t}{G_t}$$

> Where $\odot$ denotes the element-wise product.

#### 4. Reward Function
The reward $R_t$ is defined as the log-return of the portfolio value, penalized by transaction costs via the hyperparameter $\lambda$:

$$R_t = \log\left(\frac{v_t}{v_{t-1}}\right) - \lambda c_t$$
