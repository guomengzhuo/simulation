import numpy as np
import torch

class BayeHMM():
    def __init__(self, n_state_i, n_state_s, n_state_v,
                 n_observation_i, n_observation_s, n_observation_v,
                 eta, tau, sigma, d, e, c):
        np.random.seed(1024)
        torch.manual_seed(1024)
        self.n_state_i = n_state_i
        self.n_state_s = n_state_s
        self.n_state_v = n_state_v
        self.n_observation_i = n_observation_i  # c, game, G
        self.n_observation_s = n_observation_s  # e, active, A
        self.n_observation_v = n_observation_v  # d, payment, P
        # initialize Dirichlet prior
        self.eta = eta
        self.pi = np.random.dirichlet([self.eta/self.n_state_i] * self.n_state_i)  # n_state * 1
        # i^t-1 --> i^t
        self.Q_k = np.random.dirichlet([self.eta/self.n_state_i] * self.n_state_i, size=self.n_state_i)  # n_state * n_state
        # make sure sum over col to be 1
        self.tau = tau
        # i^t-1 --> s^t
        self.F_k = np.random.dirichlet([self.tau/self.n_state_s] * self.n_state_s, size=self.n_state_i)  # n_state_i * n_state_s
        # s^t-1, i^t --> s^t
        Dirichlet = torch.distributions.dirichlet.Dirichlet(torch.Tensor([self.tau/self.n_state_s] * self.n_state_s))
        self.H_uk = Dirichlet.sample([self.n_state_s, self.n_state_i]).numpy()  # dim=2, sum over dim=2 equals to 1, and size=n_state_s

        self.sigma = sigma
        self.N_u = np.random.dirichlet([self.sigma/self.n_state_v] * self.n_state_v, size=self.n_state_s)
        Dirichlet = torch.distributions.dirichlet.Dirichlet(torch.Tensor([self.sigma/self.n_state_v] * self.n_state_v))
        self.M_ou = Dirichlet.sample([self.n_state_v, self.n_state_s]).numpy()  # dim=2, sum over dim=2 equals to 1, and size=n_state_v

        self.d = d
        self.D_o = np.random.dirichlet([self.d/self.n_observation_v] * self.n_observation_v, size=self.n_state_v)
        self.e = e
        self.E_u = np.random.dirichlet([self.e/self.n_observation_s] * self.n_observation_s, size=self.n_state_s)
        self.c = c
        self.C_k = np.random.dirichlet([self.c/self.n_observation_i] * self.n_observation_i, size=self.n_state_i)

    def get_alpha(self, observation_g, observation_a, observation_p):
        n_timestamp = len(observation_g)
        alpha = []
        alpha0 = np.empty((self.n_state_i, self.n_state_s, self.n_state_v))
        #  alpha is a Tensor, dim=0: all states of i; dim=1: all states of s; dim=2: all states of v
        # dim=0
        for dim_i in range(self.n_state_i):
            for dim_s in range(self.n_state_s):
                for dim_v in range(self.n_state_v):
                    alpha0[dim_i, dim_s, dim_v] = self.pi[dim_i] * self.F_k[dim_i, dim_s] * self.N_u[dim_s, dim_v] *\
                                                  self.C_k[dim_i, observation_g[0]] * self.E_u[dim_s, observation_a[0]] * self.D_o[dim_v, observation_p[0]]

        alpha.append(alpha0)
        for t in range(1, n_timestamp):
            alpha_tmp = alpha[t - 1]
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t-1 timestamp, fixed alpha[t-1]
                        tmp = np.empty((self.n_state_i, self.n_state_s, self.n_state_v))
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp[i, s, v] =  alpha[t - 1][i, s, v] * self.M_ou[v, dim_s, dim_v] * self.H_uk[s, dim_i, dim_s] * self.Q_k[i, dim_i] * self.E_u[dim_s, observation_a[t]] * self.D_o[dim_v, observation_p[t]] * self.C_k[dim_i, observation_g[t]]
                        sum_oper = np.sum(tmp)
                        alpha_tmp[dim_i, dim_s, dim_v] = sum_oper
            alpha.append(alpha_tmp/(np.sum(alpha_tmp)+1e-5))
            # alpha.append(alpha_tmp)

        # print(len(alpha))
        return alpha

    def get_beta(self, observation_g, observation_a, observation_p):
        n_timestamp = len(observation_g)
        beta = [0] * n_timestamp
        beta[-1] = np.ones((self.n_state_i, self.n_state_s, self.n_state_v))
        beta[-1] = beta[-1] / np.sum(beta[-1])
        for t in range(n_timestamp-2, -1, -1):
            beta_tmp = beta[t+1]
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t+1 timestamp
                        tmp = np.empty((self.n_state_i, self.n_state_s, self.n_state_v))
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp[i, s, v] = beta[t+1][i, s, v] * self.Q_k[dim_i, i] * self.H_uk[dim_s, i, s] * self.M_ou[dim_v, s, v] * self.C_k[i, observation_g[t+1]] * self.E_u[s, observation_a[t+1]] * self.D_o[v, observation_p[t+1]]
                        sum_oper = np.sum(tmp)
                        beta_tmp[dim_i, dim_s, dim_v] = sum_oper
            beta[t] = beta_tmp/(np.sum(beta_tmp)+1e-5)
            # beta[t] = beta_tmp
        return beta

    def get_xi(self, alpha, beta, observation_g, observation_a, observation_p):
        n_timestamp = len(observation_g)
        xi = []
        for t in range(n_timestamp - 1):
            xi_tmp = np.empty((self.n_state_i, self.n_state_s, self.n_state_v, self.n_state_i, self.n_state_s, self.n_state_v))
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t+1 timestamp
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    xi_tmp[dim_i, dim_s, dim_v, i, s, v] = alpha[t][dim_i, dim_s, dim_v] * beta[t+1][i, s, v] * self.Q_k[dim_i, i] * self.H_uk[dim_s, i, s] * self.M_ou[dim_v, s, v] * self.C_k[i, observation_g[t+1]] * self.E_u[s, observation_a[t+1]] * self.D_o[v, observation_p[t+1]]
            xi_tmp = xi_tmp / (np.sum(xi_tmp) + 0)
            xi.append(xi_tmp)
        return xi

    def get_Q(self, old_gamma, old_xi, observation_g, observation_a, observation_p
              ): # pi, Q, H, F, M should be current iter!
        first_term = (old_gamma[0].sum(axis=(1,2)) * np.log(self.pi+1e-5)).sum()
        second_term = (old_gamma[0].sum(axis=2) * np.log(self.F_k+1e-5)).sum()
        third_term = (old_gamma[0].sum(axis=0) * np.log(self.N_u+1e-5)).sum()
        forth_term = [(t.sum(axis=(1,2,4,5))*np.log(self.Q_k+1e-5)).sum() for t in old_xi]
        forth_term = np.array(forth_term).sum()
        fifth_term = [(t.sum(axis=(0,2,5))*np.log(self.H_uk+1e-5)).sum() for t in old_xi]
        fifth_term = np.array(fifth_term).sum()
        sixth_term = [(t.sum(axis=(0,1,3))*np.log(self.M_ou+1e-5)).sum() for t in old_xi]
        sixth_term = np.array(sixth_term).sum()
        # seventh_term = [(old_gamma[t].sum(axis=(0,1))*np.log(self.D_o[:,observation_p[t]])).sum() for t in range(1, len(observation_p))]
        seventh_term = [(old_gamma[idx].sum(axis=(0,1))*np.log(self.D_o[:,t]+1e-5)).sum() for idx,t in enumerate(observation_p)]
        seventh_term = np.array(seventh_term).sum()
        eight_term = [(old_gamma[idx].sum(axis=(0,2))*np.log(self.E_u[:,t]+1e-5)).sum() for idx,t in enumerate(observation_a)]
        eight_term = np.array(eight_term).sum()
        ninth_term = [(old_gamma[idx].sum(axis=(1,2))*np.log(self.C_k[:,t]+1e-5)).sum() for idx,t in enumerate(observation_g)]
        ninth_term = np.array(ninth_term).sum()

        return first_term + second_term + third_term + forth_term + fifth_term + sixth_term + seventh_term + eight_term + ninth_term

    def update(self, gamma, xi, observation_g, observation_a, observation_p):
        # gamma[:, 0, :, :, :] --> user * i,s,v
        new_pi = (gamma[:, 0, :, :, :].sum(axis=(0, 2, 3)) + self.eta/self.n_state_i - 1) / (gamma[:, 0, :, :, :].sum() + self.eta - self.n_state_i)
        new_Q = (xi.sum(axis=(0, 1, 3, 4, 6, 7)) + self.eta/self.n_state_i - 1) / (xi.sum(axis=(0, 1, 3, 4, 5, 6, 7)) + self.eta - self.n_state_i)[:,None]
        new_F = (gamma[:, 0, :, :, :].sum(axis=(0, 3)) + self.tau/self.n_state_s -1) / (gamma[:, 0, :, :, :].sum(axis=(0, 2, 3)) + self.tau - self.n_state_s)[:, None]
        new_H = (xi.sum(axis=(0, 1, 2, 4, 7)) + self.tau/self.n_state_s - 1) / (xi.sum(axis=(0,1,2,4,6,7)) + self.tau - self.n_state_s)[:,:,None]
        new_N = (gamma[:, 0, :, :, :].sum(axis=(0,1)) + self.sigma/self.n_state_v-1) / (gamma[:, 0, :, :, :].sum(axis=(0,1,3)) + self.sigma - self.n_state_v)[:,None]
        new_M = (xi.sum(axis=(0,1,2,3,5)) + self.sigma/self.n_state_v - 1) / (xi.sum(axis=(0,1,2,3,5,7)) + self.sigma - self.n_state_v)[:,:,None]

        new_C = np.zeros((self.n_state_i, self.n_observation_i))
        new_D = np.zeros((self.n_state_v, self.n_observation_v))
        new_E = np.zeros((self.n_state_s, self.n_observation_s))
        dom_i = gamma.sum(axis=(0,1,3,4)) + self.c - self.n_observation_i
        dom_s = gamma.sum(axis=(0,1,2,4)) + self.e - self.n_observation_s
        dom_v = gamma.sum(axis=(0,1,2,3)) + self.d - self.n_observation_v

        for j in range(len(observation_g)):
            for t in range(len(observation_g[0])):
                # determine the location of each observation
                # update C, observation determines which col should be updated
                new_C[:, observation_g[j][t]] += (gamma[j, t, :, :, :].sum(axis=(1,2)) + self.c/self.n_observation_i -1)/dom_i
                new_D[:, observation_p[j][t]] += (gamma[j, t, :, :, :].sum(axis=(0,1)) + self.d/self.n_observation_v -1)/dom_v
                new_E[:, observation_a[j][t]] += (gamma[j, t, :, :, :].sum(axis=(0,2)) + self.e/self.n_observation_s -1)/dom_s

        return new_pi, new_Q, new_F, new_H, new_N, new_M, new_C, new_D, new_E

    def predict(self, test_ob_g, test_ob_a, test_ob_p):
        delta = []
        delta_0 = np.empty((self.n_state_i, self.n_state_s, self.n_state_v))
        for dim_i in range(self.n_state_i):
            for dim_s in range(self.n_state_s):
                for dim_v in range(self.n_state_v):
                    delta_0[dim_i, dim_s, dim_v] = self.pi[dim_i] * self.F_k[dim_i, dim_s] * self.N_u[dim_s, dim_v] *\
                                                  self.C_k[dim_i, test_ob_g[0]] * self.E_u[dim_s, test_ob_a[0]] * self.D_o[dim_v, test_ob_p[0]]
        delta.append(delta_0)
        for t in range(1, len(test_ob_g) - 1):
            delta_tmp = delta[t - 1]
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t-1 timestamp
                        tmp = np.empty((self.n_state_i, self.n_state_s, self.n_state_v))
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp[i, s, v] = delta[t - 1][i, s, v] * self.Q_k[i, dim_i] * self.H_uk[s, dim_i, dim_s] * self.M_ou[v, dim_s, dim_v]
                        delta_tmp[dim_i, dim_s, dim_v] = self.C_k[dim_i, test_ob_g[t]] * self.E_u[dim_s, test_ob_a[t]] * self.D_o[dim_v, test_ob_p[t]] * np.max(tmp)
            delta.append(delta_tmp)

        # At the last timestamp
        delta_T = delta[-1]/(np.sum(delta[-1]))

        # Game selection
        prob_g = []
        for g in range(self.n_observation_i):
            tmp = 0
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t-1 timestamp
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp += self.C_k[dim_i, g] * self.Q_k[i, dim_i] * self.H_uk[s, dim_i, dim_s] * self.M_ou[v, dim_s, dim_v] * delta_T[i, s, v]
            prob_g.append(tmp)

        # Active time
        prob_a = []
        for a in range(self.n_observation_s):
            tmp = 0
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t-1 timestamp
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp += self.E_u[dim_s, a] * self.Q_k[i, dim_i] * self.H_uk[s, dim_i, dim_s] * self.M_ou[v, dim_s, dim_v] * delta_T[i, s, v]
            prob_a.append(tmp)

        # Payement
        prob_p = []
        for p in range(self.n_observation_v):
            tmp = 0
            # for t timestamp
            for dim_i in range(self.n_state_i):
                for dim_s in range(self.n_state_s):
                    for dim_v in range(self.n_state_v):
                        # for t-1 timestamp
                        for i in range(self.n_state_i):
                            for s in range(self.n_state_s):
                                for v in range(self.n_state_v):
                                    tmp += self.D_o[dim_v, p] * self.Q_k[i, dim_i] * self.H_uk[s, dim_i, dim_s] * \
                                           self.M_ou[v, dim_s, dim_v] * delta_T[i, s, v]
            prob_p.append(tmp)

        return prob_g, prob_a, prob_p
