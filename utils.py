import numpy as np
import json
import tool4plot

def BICandAIC(i, s, v, c, e, d, ll, N):
    numPara = i-1 + s*(i-1) + v*(s-1)+i*(i-1) + i*s*(s-1) + s*v*(v-1) + i*(c-1) + s*(e-1) + v*(d-1)
    bic = -2.0*ll + np.log(N)*numPara
    aic = -2.0*ll + 2*numPara
    return aic, bic

def auc(prob_g, prob_a, prob_p, true_g, true_a, true_p):
    from sklearn.metrics import roc_auc_score,average_precision_score
    import numpy as np
    rank_g = 1 - np.array(prob_g)[:,0]
    rank_a = 1 - np.array(prob_a)[:,0]
    rank_p = 1 - np.array(prob_p)[:,0]
    true_g = np.array(true_g).astype(int)
    true_a = np.array(true_a).astype(int)
    true_p = np.array(true_p).astype(int)

    auc_1 = roc_auc_score(true_g, rank_g.astype(float))
    auc_2 = roc_auc_score(true_a, rank_a.astype(float))
    auc_3 = roc_auc_score(true_p, rank_p.astype(float))

    pr_auc_1 = average_precision_score(true_g, rank_g.astype(float))
    pr_auc_2 = average_precision_score(true_a, rank_a.astype(float))
    pr_auc_3 = average_precision_score(true_p, rank_p.astype(float))

    return auc_1, auc_2, auc_3, pr_auc_1, pr_auc_2, pr_auc_3

def plotMatrix(args,json_file):
    model_name = 'i_' + str(args.n_state_i) + '_s_' + str(args.n_state_s) + '_v_' + str(args.n_state_v)
    file = open(json_file, 'r')
    result = json.load(file)
    file.close()
    # print(result[model_name])

    # plot initial distributions
    true_pi = np.array(result[model_name]['true_pi'])
    pred_pi = np.array(result[model_name]['pi'])
    tool4plot.barPlot(true_pi, pred_pi)

    # plot transition matrices
    data = {
        'pred_Q': np.array(result[model_name]['Q_k']),
        'pred_F': np.array(result[model_name]['F_k']),
        'pred_N': np.array(result[model_name]['N_u']),
        'true_Q': np.array(result[model_name]['true_Q_k']),
        'true_F': np.array(result[model_name]['true_F_k']),
        'true_N': np.array(result[model_name]['true_N_u']),
    }
    tool4plot.tranMat(data)

    # plot s->s v->v
    data = {
        'pred_M': np.array(result[model_name]['M_ou']),
        'pred_H': np.array(result[model_name]['H_uk']),
        'true_M': np.array(result[model_name]['true_M_ou']),
        'true_H': np.array(result[model_name]['true_H_uk']),
    }
    tool4plot.plotMat2(data)

    data = {
        'pred_C': np.array(result[model_name]['C_k']),
        'pred_D': np.array(result[model_name]['D_o']),
        'pred_E': np.array(result[model_name]['E_u']),
        'true_C': np.array(result[model_name]['true_C_k']),
        'true_D': np.array(result[model_name]['true_D_o']),
        'true_E': np.array(result[model_name]['true_E_u']),
    }
    tool4plot.plotObser(data)


