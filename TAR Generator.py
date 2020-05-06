
"""
Threshhold Autoregressive Process (TAR) Generator 
"""
import numpy as np
import matplotlib.pyplot as plt

class TAR:
    def __init__(self,
                        mu_params =     np.array( [0.05, 0.05]) ,
                        
                        sigma_params =  np.array( [0.25, 0.25]),
                        
                        AR_params =     np.array([[0.50, 0 ],
                                                 [-0.50, 0] ])   
                        ):
        
        """
            mu_params = Expected return for each state
            sigma_params = Expected volatility for each state
            AR_params = Autoregressive parameters for each state
        """
        self.mu_params, self.sigma_params, self.AR_params = mu_params, sigma_params, AR_params 
        self.k = len(mu_params)
        self.r = None
        self.y = None
    
    def sim_transitionvar(self, N=750, threshold=0):
        e = np.random.randn(N)
        transvar = np.cumsum(e)
        regime = np.where(transvar>threshold,0,1)
    
        return regime, transvar
    
    
    def sim(self, N=750, state_0 = 0):
        dt = 1/250
        # sim shocks
        e = np.random.randn(N)   # sim shocks
        
        # set first observation
        e[0:2] = e[0:2] * self.sigma_params[state_0] * np.sqrt(dt) 
        r = e.copy()
        r[0:2] = r[0:2] + self.mu_params[state_0] * dt
           
        # sim regime states (and transition variable)
        self.regime, self.transvar = self.sim_transitionvar(N)
    
            
        # Simulate:
        for t in np.arange(1,N):
            # determine state
            state = self.regime[t]
            # calc returns for given state
            e[t] = e[t] * self.sigma_params[state] * np.sqrt(dt)
            mu = self.mu_params[state] * dt + e[t]
            r[t] = mu + e[t] + self.AR_params[state,0]*r[t-1] + self.AR_params[state,1]*r[t-2]
        
        self.r = r
        self.y = 10*np.exp(np.cumsum(r))
       
    

    def plot_sim(self, colored=True):
        "Plot generated data"
        plt.style.use('ggplot')
        r = self.r
        regime = self.regime
        transvar = self.transvar
        y = self.y
        fig, axes = plt.subplots(3, figsize=(10,12))
        
        # ax 0
        ax = axes[0]
        ax.plot(transvar, 'k', linewidth = .7)
        ax.margins(x=0)
        ax.hlines(0,0,len(r), linestyles='dashed')
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(transvar), max(transvar),# where=transvar<=0,
                            facecolor='green', alpha = .3)
            ax.fill_between(np.arange(len(r)), min(transvar), max(transvar), where=transvar>=0,
                            facecolor='blue', alpha = .3)
        ax.set(title='Exogenious Transition Variable')
        
        
        
        # ax 1
        ax = axes[1]
        ax.plot(r, 'k', linewidth = .7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(r), max(r), #where=transvar<=0,
                            facecolor='green', alpha = .3)
            ax.fill_between(np.arange(len(r)), min(r), max(r),  where=transvar>=0, 
                            facecolor='blue', alpha = .3)
        ax.set(title='Simulated Returns')
        
        # ax 2
        ax = axes[2]
        ax.plot(y, 'k', linewidth = .7)
        ax.margins(x=0)
        if colored == True:
            ax.fill_between(np.arange(len(r)), min(y), max(y),# where=transvar<=0, 
                            facecolor='green', alpha = .3)
            ax.fill_between(np.arange(len(r)), min(y), max(y), where=transvar>=0,
                            facecolor='blue', alpha = .3)

        ax.set(title='Simulated Prices')
        ax.set_yscale('log')
        plt.show()



if __name__ == "__main__":
    N = 250
    mu_params =     np.array( [0.05, 0.05])
                            
    sigma_params =  np.array( [0.25, 0.25])
                            
    AR_params =     np.array([[0.50, 0 ],
                              [-0.50, 0] ])   
    
    tar_model = TAR(mu_params, sigma_params, AR_params)
    
    tar_model.sim(N)
    tar_model.plot_sim(colored=True)
