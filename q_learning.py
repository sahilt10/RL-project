import numpy as np
from env import MultiStockEnv

def simulate_agent(price_data, episodes=300):
    env = MultiStockEnv(price_data)
    q_table = {}

    alpha = 0.05
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    for ep in range(episodes):
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        state = env.reset()
        done = False

        while not done:
            state_key = tuple(np.round(state, 2))
            if state_key not in q_table:
                q_table[state_key] = np.zeros((env.n_stocks, 3))  # hold, buy, sell

            actions = []
            for i in range(env.n_stocks):
                if np.random.rand() < epsilon:
                    actions.append(np.random.choice(3))
                else:
                    actions.append(np.argmax(q_table[state_key][i]))

            next_state, reward, done = env.step(actions)
            next_state_key = tuple(np.round(next_state, 2))

            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros((env.n_stocks, 3))

            for i in range(env.n_stocks):
                old_q = q_table[state_key][i][actions[i]]
                next_max = np.max(q_table[next_state_key][i])
                q_table[state_key][i][actions[i]] = old_q + alpha * (reward + gamma * next_max - old_q)

            state = next_state

    return env.portfolio_values, env.transactions
