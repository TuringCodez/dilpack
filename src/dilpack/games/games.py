import numpy as np
import itertools


class RockPaperScissors:
    def __init__(self, x=None, y=None):
        self.payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        self.x = np.array(x) if x is not None else np.array([1 / 3, 1 / 3, 1 / 3])
        self.y = np.array(y) if y is not None else np.array([1 / 3, 1 / 3, 1 / 3])

    def calculate_expected_payoff(self):
        return self.x.T @ self.payoff_matrix @ self.y


class BasketballOffense:
    def __init__(self, x=None, y=None):
        self.payoff_matrix = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.x = np.array(x) if x is not None else np.array([20, 20, 20])
        self.y = np.array(y) if y is not None else np.array([0.362, 0.4515, 0.394])

    def calculate_expected_payoff(self, x=None):
        shots_dist = x if x is not None else self.x
        return shots_dist @ self.payoff_matrix @ self.y

    def optimal_strategy(self):
        shots_grid = dict(
            zip(["3PT", "MR", "UB"], [np.arange(0, 61, 1) for _ in range(3)])
        )

        a = shots_grid.values()
        combinations = list(itertools.product(*a))
        combinations = [x for x in combinations if sum(x) == 60]

        max_exp = 0
        index = 0
        for i in range(len(combinations)):
            tmp = self.calculate_expected_payoff(np.array(combinations[i]))
            if tmp > max_exp:
                max_exp = tmp
                index = i

        return combinations[index]


class NBASportsBetting:
    def __init__(self, money=None, odds=None, win_pct=None):
        self.money = money if money is not None else 100
        self.odds = odds if odds is not None else [400, 550, 800, 900]
        self.payoff_matrix = self.generate_payoff()
        self.teams = ["Celtics", "Bucks", "Nets", "Nuggets"]
        self.win_pct = (
            win_pct
            if win_pct is not None
            else np.array([0.714, 0.659, 0.675, 0.683, 0.125])
        )
        self.probs = (self.win_pct / self.win_pct.sum()).reshape(-1, 1)

    def generate_payoff(self):
        p = np.diag(self.odds)
        p = np.c_[p, np.zeros(4)]
        p[p == 0] = self.money

        return p

    def calculate_expected_payoff(self, x):
        money_to_bet = np.array(x) / 100
        return (money_to_bet @ self.payoff_matrix @ self.probs)[0]

    def optimal_strategy(self):
        credits_grid = dict(
            zip(
                self.teams,
                [np.arange(0, self.money + 100, 100) for _ in range(len(self.teams))],
            )
        )

        a = credits_grid.values()
        combinations = list(itertools.product(*a))
        combinations = [x for x in combinations if sum(x) == self.money]

        max_exp = 0
        index = 0
        for i in range(len(combinations)):
            tmp = self.calculate_expected_payoff(np.array(combinations[i]))
            if tmp > max_exp:
                max_exp = tmp
                index = i

        return combinations[index]
