"""
Threshhold Autoregressive Process (TAR) Generator 
"""
import numpy as np
import matplotlib.pyplot as plt


class STAR:
    def __init__(self,
                 mu_params=np.array([0.05, 0.05]),

                 sigma_params=np.array([0.25, 0.25]),

                 AR_params=np.array([[0.50, 0],
                                     [-0.50, 0]]),
                 gamma=float('inf'),
                 c=0.0
                 ):
        """
            mu_params = Expected return for each state
            sigma_params = Expected volatility for each state
            AR_params = Autoregressive parameters for each state
        """
        self.gamma, self.c, self.mu_params, self.sigma_params, self.AR_params = gamma, c, mu_params, sigma_params, AR_params
        self.k = len(mu_params)
        self.r = None
        self.y = None

    def cumsum_with_limits(self, values, lower_limit=-1, upper_limit=1):
        n = len(values)
        new_values = np.empty(n)
        sum_val = 0
        for i in range(n):
            x = max(min(sum_val + values[i],
                    upper_limit), lower_limit) - sum_val
            new_values[i] = x
            sum_val += x
        res = np.cumsum(new_values)
        return res

    def G(self, s, c=0.0, gamma=float('inf')):
        """
         Transition Function G:
         s = transition variable
         c = threshold parameter
         gamma = steepness parameter

        """
        output = (1 + np.exp(-gamma * (s - c)))**(-1)
        return output

    def sim_transitionvar(self, N=750, threshold=0):
        # np.random.seed(1)
        e = np.random.randn(N)
        transvar = self.cumsum_with_limits(e, lower_limit=-4, upper_limit=4)
        regime = self.G(transvar, c=self.c, gamma=self.gamma)
        return regime, transvar

    def sim(self, N=750, state_0=0):
        dt = 1 / 250
        # sim shocks
        e = np.random.randn(N)   # sim shocks

        # set first observation
        e[0:2] = e[0:2] * self.sigma_params[state_0] * np.sqrt(dt)
        r = e.copy()
        r[0:2] = r[0:2] + self.mu_params[state_0] * dt

        # sim regime states (and transition variable)
        self.regime, self.transvar = self.sim_transitionvar(N)

        # Simulate:
        for t in np.arange(2, N):
            # determine state
            state = self.regime[t]
            # calc returns for given state
            e[t] = (state * (e[t] * self.sigma_params[0]) + (1 - state)
                    * (e[t] * self.sigma_params[1])) * np.sqrt(dt)
            mu = (state * self.mu_params[0] +
                  (1 - state) * self.mu_params[1]) * dt
            r[t] = mu + e[t] + state * (self.AR_params[0, 0] * r[t - 1] + self.AR_params[0, 1] * r[t - 2]) + (
                1 - state) * (self.AR_params[1, 0] * r[t - 1] + self.AR_params[1, 1] * r[t - 2])
        self.r = r
        self.y = 100 * np.exp(np.cumsum(r))

    def plot(self, colored=True):
        # plt.style.use('ggplot')
        r = self.r
        regime = self.regime
        transvar = self.transvar
        y = self.y
        fig, axes = plt.subplots(3, figsize=(10, 12))

        # ax 0
        ax = axes[0]
        ax.plot(transvar, 'k', linewidth=.7)
        # ax.plot(self.regime)
        ax.margins(x=0)
        ax.hlines(self.c, 0, len(r), linestyles='dashed')
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(transvar), max(transvar),
                            facecolor='green', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(transvar), max(transvar), where=transvar >= 0,
                            facecolor='blue', alpha=.3)
        ax.set(title='Exogenous Transition Variable')

        # ax 1
        ax = axes[1]
        ax.plot(r, 'k', linewidth=.7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(r), max(r),
                            facecolor='green', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(r), max(r), where=transvar >= 0,
                            facecolor='blue', alpha=.3)
        ax.set(title='Simulated Returns')

        # ax 2
        ax = axes[2]
        ax.plot(y, 'k', linewidth=.7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(y), max(y),
                            facecolor='green', alpha=.3)
            ax.fill_between(np.arange(len(r)), min(y), max(y), where=transvar >= 0,
                            facecolor='blue', alpha=.3)

        ax.set(title='Simulated Prices')
        # ax.set_yscale('log')
        plt.show()
