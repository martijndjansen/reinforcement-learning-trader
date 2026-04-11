# notebook-reinforcement-learning



## Improvements
1. CNN or LSTM over MLP
2. Add new features like:
    1. Economic variables
    2. Removing full price features?
    3. More stocks
3. Implement Differential Sharpe Ratio
    $$R_t = DSR_t - \lambda * (Transaction costs)_t$$
4. Dirichlet distribution implementation

## Implementation
1. PPO + Dirichlet. PPO over SAC as SAC does not have Dirichlet 'support'
