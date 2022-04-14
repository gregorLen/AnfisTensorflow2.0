# test_star_process
from star_process import STAR
import numpy as np

if __name__ == "__main__":
    N = 250
    mu_params = np.array([0.05, -0.15])

    sigma_params = np.array([0.0, 0.0])

    AR_params = np.array([[0.0, 0],
                          [0.0, 0]])

    #gamma = 1.5
    gamma = float('inf')

    star_model = STAR(mu_params, sigma_params, AR_params, gamma)

    star_model.sim(N)
    star_model.plot(colored=True)
