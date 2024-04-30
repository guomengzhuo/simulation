import numpy as np
import logging
import sys
import argparse
import json
import data_generate_process
import model
from sklearn.model_selection import train_test_split
from utils import auc, BICandAIC, plotMatrix


logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


def train(HMM, delta, observation_g, observation_a, observation_p):
    Q_old = -10000000.0
    Q_new = 0
    iter = 0
    while abs((Q_new - Q_old) / Q_old) > delta:
        logging.info("------------This is current {} iteration------------".format(iter))
        iter += 1
        alpha = []  # user * timestamp * (i, s, v)
        beta = []  # user * timestamp * (i,s,v)
        gamma = []  # user * timestamp * (i,s,v)
        xi = []  # user * timestamp-1 * (i,s,v,i,s,v)
        Q = []  # user
        for idx in range(len(observation_g)):
            # HMM.fix_prob_trans()
            alpha_j = HMM.get_alpha(observation_g[idx], observation_a[idx],
                                    observation_p[idx])  # timestamp * (i,s,v)
            alpha.append(alpha_j)
            beta_j = HMM.get_beta(observation_g[idx], observation_a[idx], observation_p[idx])
            beta.append(beta_j)
            # get gamma
            gamma_j = []
            for t in range(len(alpha_j)):
                # print(np.sum(alpha_j[t] * beta_j[t]))
                gamma_j.append(alpha_j[t] * beta_j[t] / np.sum(alpha_j[t] * beta_j[t]))
            gamma.append(gamma_j)

            # get xi
            xi_j = HMM.get_xi(alpha_j, beta_j, observation_g[idx], observation_a[idx], observation_p[idx])
            xi.append(xi_j)
            # get Q

            Q_j = HMM.get_Q(gamma_j, xi_j, observation_g[idx], observation_a[idx], observation_p[idx])
            Q.append(Q_j)

        # first iteration
        if iter == 1:
            Q_old = np.array(Q).sum()
        else:
            Q_old = Q_new
            Q_new = np.array(Q).sum()
        # update
        HMM.pi, HMM.Q_k, HMM.F_k, HMM.H_uk, HMM.N_u, HMM.M_ou, HMM.C_k, HMM.D_o, HMM.E_u = HMM.update(
            np.array(gamma),
            np.array(xi),
            observation_g,
            observation_a,
            observation_p)
    return Q_new

def predict(HMM, tst_observation_g, tst_observation_a, tst_observation_p):
    # prediction
    rec_g = []
    rec_a = []
    rec_p = []
    true_g = []
    true_a = []
    true_p = []
    for idx in range(len(tst_observation_g)):
        prob_g, prob_a, prob_p = HMM.predict(tst_observation_g[idx], tst_observation_a[idx],
                                             tst_observation_p[idx])
        rec_g.append(prob_g)
        true_g.append(tst_observation_g[idx][-1])
        rec_a.append(prob_a)
        true_a.append(tst_observation_a[idx][-1])
        rec_p.append(prob_p)
        true_p.append(tst_observation_p[idx][-1])

    auc_g, auc_a, auc_p, pr_g, pr_a, pr_p = auc(rec_g, rec_a, rec_p, true_g, true_a, true_p)
    logging.info(
        'auc_g: {} auc_a:{} auc_p:{} pr_g:{} pr_a:{} pr_p:{}'.format(auc_g, auc_a,
                                                                     auc_p,
                                                                     pr_g, pr_a, pr_p))
    return auc_g, auc_a, auc_p, pr_g, pr_a, pr_p

def save_dicts(HMM, DGP, model_name, metric, file_name='result_dict_sim.json'):
    result_dict = {}
    if model_name not in result_dict.keys():
        result_dict[model_name] = {}
        result_dict[model_name]['metric'] = metric
        # save predicted mat
        result_dict[model_name]['pi'] = HMM.pi.tolist()
        result_dict[model_name]['Q_k'] = HMM.Q_k.tolist()
        result_dict[model_name]['F_k'] = HMM.F_k.tolist()
        result_dict[model_name]['H_uk'] = HMM.H_uk.tolist()
        result_dict[model_name]['N_u'] = HMM.N_u.tolist()
        result_dict[model_name]['M_ou'] = HMM.M_ou.tolist()
        result_dict[model_name]['C_k'] = (np.ones_like(HMM.C_k) - HMM.C_k).tolist()
        result_dict[model_name]['D_o'] = (np.ones_like(HMM.D_o) - HMM.D_o).tolist()
        result_dict[model_name]['E_u'] = (np.ones_like(HMM.E_u) - HMM.E_u).tolist()

        # save true mat
        result_dict[model_name]['true_pi'] = DGP.pi.tolist()
        result_dict[model_name]['true_Q_k'] = DGP.Q_prior.tolist()
        result_dict[model_name]['true_F_k'] = DGP.F_prior.tolist()
        result_dict[model_name]['true_H_uk'] = DGP.H_prior.tolist()
        result_dict[model_name]['true_N_u'] = DGP.N_prior.tolist()
        result_dict[model_name]['true_M_ou'] = DGP.M_prior.tolist()
        result_dict[model_name]['true_C_k'] = DGP.C_prior.tolist()
        result_dict[model_name]['true_D_o'] = DGP.D_prior.tolist()
        result_dict[model_name]['true_E_u'] = DGP.E_prior.tolist()

    # save
    json_str = json.dumps(result_dict, indent=3)
    with open('result_dict_sim.json', 'w') as js:
        js.write(json_str)

def main(args):
    logging.info("------------Start DGP------------")
    dgp = data_generate_process.simulationData(
        args.N,
        args.nState,
        args.nObser,
        args.seed,
        args.T
    )
    # simulation data
    X_train, X_test = train_test_split(dgp.X, test_size=.2, random_state=45)

    logging.info("------------Done DGP------------")
    observation_g = np.array(X_train)[:, 0, :]
    observation_a = np.array(X_train)[:, 1, :]
    observation_p = np.array(X_train)[:, 2, :]

    tst_observation_g = np.array(X_test)[:, 0, :]
    tst_observation_a = np.array(X_test)[:, 1, :]
    tst_observation_p = np.array(X_test)[:, 2, :]

    HMM = model.BayeHMM(
        n_state_i=args.n_state_i,
        n_state_s=args.n_state_s,
        n_state_v=args.n_state_v,
        n_observation_i=args.nObser[0],
        n_observation_s=args.nObser[1],
        n_observation_v=args.nObser[2],
        eta=args.n_state_i,
        tau=args.n_state_s,
        sigma=args.n_state_v,
        d=args.nObser[2],
        e=args.nObser[1],
        c=args.nObser[0]
    )
    Q_new = train(HMM, 0.01, observation_g, observation_a, observation_p)
    auc_g, auc_a, auc_p, pr_g, pr_a, pr_p = predict(HMM, tst_observation_g, tst_observation_a, tst_observation_p)

    logging.info(
        "--------------Caculate BIC and AIC------------------")

    aic, bic = BICandAIC(args.n_state_i,
                         args.n_state_s,
                         args.n_state_v,
                         args.nObser[0],
                         args.nObser[1],
                         args.nObser[2],
                         Q_new,
                         len(observation_g))
    logging.info(
        "--------------aic: {} bic:{}--------------".format(
            aic, bic))

    logging.info(
        "--------------Save Results------------------")
    model_name = 'i_' + str(args.n_state_i) + '_s_' + str(args.n_state_s) + '_v_' + str(args.n_state_v)
    metric = [auc_g, auc_a, auc_p, pr_g, pr_a, pr_p, aic, bic]
    save_dicts(HMM, dgp, model_name, metric)

    logging.info(
        "--------------Start Plotting------------------")
    # Plotting


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--N", default=1500, type=int)
    parser.add_argument("-nState", "--nState", default=[2,2,2], type=list)
    parser.add_argument("-nObser", "--nObser", default=[2,2,2], type=list)
    parser.add_argument("-T", "--T", default=10, type=int)
    parser.add_argument("-seed", "--seed", default=1024, type=int)
    parser.add_argument("-i", "--n-state-i", default=2, type=int)
    parser.add_argument("-s", "--n-state-s", default=2, type=int)
    parser.add_argument("-v", "--n-state-v", default=2, type=int)
    args = parser.parse_args()
    main(args)
    plotMatrix(args, json_file='result_dict_sim.json')
