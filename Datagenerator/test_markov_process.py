# test_markov_process.py
from markov_process import MRS
import numpy as np

if __name__ == "__main__":
    N = 1000
    mu_params = np.array([0.07, 0.0, -0.55])
    sigma_params = np.array([.1, .25, .60])
    P = np.array([[0.989, 0.01, 0.001],        # Transition Matrix
                  [0.03, 0.969, 0.001],
                  [0.00, 0.03, 0.97]])
    AR_params = np.array([[0.4, -0.2],
                          [0.5, -0.3],
                          [0.8, -.4]])

    mrs_model = MRS(P, mu_params, sigma_params, AR_params)

    mrs_model.sim(N)
    mrs_model.plot(colored=True)
