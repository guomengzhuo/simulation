import utils
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import json
import argparse
import logging
import sys
from tqdm import tqdm

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


class simulationData():
    def __init__(self, N, nState, nObser, seed, T):
        assert len(nState) == len(nObser)
        self.X = []
        self.N = N
        self.nState = nState
        self.nObser = nObser
        self.seed = seed
        self.T = T
        # define transiftion matrices
        self.pi = np.array([1 / nState[0]] * nState[0])
        self.F_prior = np.array(
            [[0.95, 0.05],
             [0.05, 0.95],
             ])
        self.N_prior = np.array(
            [[0.85, 0.15],
             [0.15, 0.85], ]
        )
        self.Q_prior = np.array(
            [[0.95, 0.05],
             [0.05, 0.95], ]
        )
        self.H_prior = np.array(
            [[[0.95, 0.05],
              [0.85, 0.15], ],

             [[0.25, 0.75],
              [0.05, 0.95], ]]
        )
        self.M_prior = np.array(
            [[[0.95, 0.05],
              [0.85, 0.15], ],

             [[0.25, 0.75],
              [0.05, 0.95], ]]
        )
        self.C_prior = np.array(
            [[0.9, 0.1],
             [0.1, 0.9]]
        )
        self.E_prior = np.array(
            [[0.9, 0.1],
             [0.1, 0.9]]
        )
        self.D_prior = np.array(
            [[0.95, 0.05],
             [0.35, 0.65]]
        )
        self.dgp()

    def dgp(self):
        # data generation process
        for n in range(self.N):
            np.random.seed(self.seed + n)
            x_g = []
            x_a = []
            x_p = []
            for t in range(self.T):
                if t == 0:
                    # determine state and observation
                    temp_state_i = np.random.multinomial(n=1, pvals=self.pi).argmax()
                    C = self.C_prior[temp_state_i]
                    obser_g = np.random.multinomial(n=1, pvals=C).argmax()
                    x_g.append(obser_g)

                    temp_state_s = np.random.multinomial(n=1, pvals=self.F_prior[temp_state_i]).argmax()
                    E = self.E_prior[temp_state_s]
                    obser_a = np.random.multinomial(n=1, pvals=E).argmax()
                    x_a.append(obser_a)

                    temp_state_v = np.random.multinomial(n=1, pvals=self.N_prior[temp_state_s]).argmax()
                    D = self.D_prior[temp_state_v]
                    obser_p = np.random.multinomial(n=1, pvals=D).argmax()
                    x_p.append(obser_p)

                if t >= 1:
                    # determine current state given the last state
                    current_state_i = np.random.multinomial(n=1, pvals=self.Q_prior[temp_state_i]).argmax()
                    current_state_s = np.random.multinomial(n=1,
                                                            pvals=self.H_prior[temp_state_i][temp_state_s]).argmax()
                    current_state_v = np.random.multinomial(n=1,
                                                            pvals=self.M_prior[temp_state_s][temp_state_v]).argmax()

                    # determine observation
                    C = self.C_prior[current_state_i]
                    obser_g = np.random.multinomial(n=1, pvals=C).argmax()
                    x_g.append(obser_g)

                    E = self.E_prior[current_state_s]
                    obser_a = np.random.multinomial(n=1, pvals=E).argmax()
                    x_a.append(obser_a)

                    D = self.D_prior[current_state_v]
                    obser_p = np.random.multinomial(n=1, pvals=D).argmax()
                    x_p.append(obser_p)

                    temp_state_i = current_state_i
                    temp_state_s = current_state_s
                    temp_state_v = current_state_v

            self.X.append([x_g, x_a, x_p])

