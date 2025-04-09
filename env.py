import numpy as np

class MultiStockEnv:
    def __init__(self, price_data):
        self.price_data = price_data
        self.n_steps = len(price_data)
        self.n_stocks = price_data.shape[1]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.holdings = np.zeros(self.n_stocks)
        self.transactions = []
        self.portfolio_values = []
        return self._get_state()

    def _get_state(self):
        step = min(self.current_step, self.n_steps - 1)
        norm_prices = self.price_data.iloc[step] / self.price_data.iloc[0]
        return np.concatenate([[self.balance], self.holdings, norm_prices])

    def _get_portfolio_value(self, prices):
        return self.balance + np.dot(self.holdings, prices)

    def step(self, actions):
        step = min(self.current_step, self.n_steps - 1)
        prices = self.price_data.iloc[step]

        for i, action in enumerate(actions):
            if action == 1 and self.balance >= prices.iloc[i]:
                self.holdings[i] += 1
                self.balance -= prices.iloc[i]
                self.transactions.append((self.current_step, self.price_data.columns[i], "BUY", prices.iloc[i]))
            elif action == 2 and self.holdings[i] > 0:
                self.holdings[i] -= 1
                self.balance += prices.iloc[i]
                self.transactions.append((self.current_step, self.price_data.columns[i], "SELL", prices.iloc[i]))

        portfolio_val = self._get_portfolio_value(prices)
        self.portfolio_values.append(portfolio_val)

        self.current_step += 1
        done = self.current_step >= self.n_steps
        return self._get_state(), portfolio_val, done
