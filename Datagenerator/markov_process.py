"""
This module simulates a Markov Regime-Switching process
"""
import numpy as np
import matplotlib.pyplot as plt


class MRS:
    def __init__(self,
                 P=np.array([[0.989, 0.01, 0.001],  # Transition Matrix
                             [0.03, 0.969, 0.001],
                             [0.00, 0.03, 0.97]]),

                 mu_params=np.array([0.07, 0.0, -0.55]),

                 sigma_params=np.array([.1, .25, .60]),

                 AR_params=np.array([[0.4, -0.2],
                                     [0.5, -0.3],
                                     [0.8, -.4]])
                 ):
        """
            P = Transition Matrix
            mu_params = Expected return for each state
            sigma_params = Expected volatility for each state
            AR_params = Autoregressive parameters for each state
        """
        self.P, self.mu_params, self.sigma_params, self.AR_params = P, mu_params, sigma_params, AR_params

        check = P.shape[0] == P.shape[1] == len(
            mu_params) == len(sigma_params) == AR_params.shape[0]
        if check == False:
            raise ValueError('Dimensions of parameters does not fit!')

        self.markovchain = []
        self.k = len(mu_params)
        self.r = None
        self.y = None

    def roulettewheel(self, prob):
        "Update state"
        cp = np.cumsum(prob) / np.sum(prob)
        u = np.random.rand(1)
        i = 0
        while u > cp[i]:
            i += 1
        return i

    def sim(self, N, state_0=0):
        "Simulate a Markov Regime Switching time series of length N"
        dt = 1 / 250
        e = np.random.randn(N)   # sim shocks
        state = state_0
        e[0:2] = e[0:2] * self.sigma_params[state] * np.sqrt(dt)
        self.r = e.copy()
        self.r[0:2] = self.r[0:2] + self.mu_params[state] * dt
        self.markovchain = np.repeat(state, 2).astype(int)

        # Simulate:
        for t in np.arange(2, N):
            # determine state
            state = self.roulettewheel(self.P[state])
            self.markovchain = np.append(self.markovchain, state)
            # calc returns for given state
            e[t] = e[t] * self.sigma_params[state] * np.sqrt(dt)
            mu = self.mu_params[state] * dt  # + e[t]
            self.r[t] = mu + e[t] + self.AR_params[state, 0] * \
                self.r[t - 1] + self.AR_params[state, 1] * self.r[t - 2]
        self.y = 10 * np.exp(np.cumsum(self.r))

    def plot(self, colored=True):
        "Plot generated data"
        plt.style.use('ggplot')
        r = self.r
        mc = self.markovchain
        y = self.y
        fig, axes = plt.subplots(2, figsize=(10, 6))

        ax = axes[0]
        ax.plot(r, 'k', linewidth=.7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(r), max(
                r), where=mc >= 0, facecolor='green', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(r), max(
                r), where=mc >= 1, facecolor='yellow', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(r), max(
                r), where=mc >= 2, facecolor='red', alpha=.3)
        ax.set(title='Simulated Returns')

        ax = axes[1]
        ax.plot(y, 'k', linewidth=.7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(y), max(y), where=mc >= 0,
                            facecolor='green', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(y), max(y), where=mc >= 1,
                            facecolor='yellow', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(y), max(y), where=mc >= 2,
                            facecolor='red', alpha=.3)
        ax.set(title='Simulated Prices')
        ax.set_yscale('log')
        plt.show()
