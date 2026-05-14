# Simulation4BayesHMM
 This is the demo code for reproducing the simulation results.
 
### Requirements
Python == 3.7.   
Sklearn:0.21.3, Numpy: 1.19.5, Pandas: 0.25.1, Matplotlib: 3.5.3   
All the codes are run on the CPU by default. 

### Reproduce Figures in the Appendix
```
python main.py
```


# Simulation Results

This section describes the synthetic data generating process and the interpretation of predefined transition matrices for the proposed three-layered dynamic hidden Markov model (3DHMM).


## Synthetic Data Generating Process

The goal of the simulation study is to evaluate whether the proposed model can recover predefined transition patterns and correctly capture user dynamic behavior in a mobile gaming context.

We define specialized transition matrices based on assumptions about users' behavioral decision journeys. These matrices are used to generate synthetic behavioral sequences. The generated data are then used to train the 3DHMM and evaluate whether the learned transition matrices are consistent with the predefined true matrices.

### Model Inputs

The synthetic data generating process uses the following predefined probabilities:

| Notation | Description |
|---|---|
| $\pi_k=P(i=k)$ | Initial probability of goal-orientation state $i$ |
| $F_{ku}=P(s=u\mid i=k)$ | Initial probability of engagement state $s$ conditional on goal-orientation state $i$ |
| $N_{uo}=P(v=o\mid s=u)$ | Initial probability of purchase-intention state $v$ conditional on engagement state $s$ |
| $Q_{kk'}=P(i'=k'\mid i=k)$ | Transition probability of goal-orientation states |
| $H_{uk'u'}=P(s'=u'\mid i'=k',s=u)$ | Transition probability of engagement states conditional on current goal-orientation |
| $M_{ou'o'}=P(v'=o'\mid s'=u',v=o)$ | Transition probability of purchase-intention states conditional on current engagement |
| $C_{kg}=P(g\mid i=k)$ | Observation probability of game preference $g$ conditional on goal-orientation |
| $E_{ua}=P(a\mid s=u)$ | Observation probability of activity $a$ conditional on engagement |
| $D_{op}=P(p\mid v=o)$ | Observation probability of purchase $p$ conditional on purchase intention |

In the simulation, all latent states and observations are binary:

$$
k,u,o,k',u',o',g,a,p \in \{1,2\}.
$$

The process also requires the number of samples $N$ and the length of each sequence $T$.

## Algorithm

```text
Input:
    Predefined transition matrices:
        pi, F, N, Q, H, M, C, E, D
    Number of samples:
        N
    Length of each sample:
        T

For j = 1 to N:
    For t = 1 to T:
        If t = 1:
            Sample goal-orientation state:
                i_j^t ~ pi

            Generate game-preference observation:
                g_j^t ~ C_{i_j^t,g} = P(g | i_j^t)

            Sample engagement state:
                s_j^t ~ F_{i_j^t,s} = P(s | i_j^t)

            Generate activity observation:
                a_j^t ~ E_{s_j^t,a} = P(a | s_j^t)

            Sample purchase-intention state:
                v_j^t ~ N_{s_j^t,v} = P(v | s_j^t)

            Generate purchase observation:
                p_j^t ~ D_{v_j^t,p} = P(p | v_j^t)

        Else:
            Sample goal-orientation state:
                i_j^t ~ Q_{i_j^{t-1},i} = P(i_j^t | i_j^{t-1})

            Sample engagement state:
                s_j^t ~ H_{s_j^{t-1},i_j^t,s} = P(s | i_j^t, s_j^{t-1})

            Sample purchase-intention state:
                v_j^t ~ M_{v_j^{t-1},s_j^t,v} = P(v | s_j^t, v_j^{t-1})

            Generate game-preference observation:
                g_j^t ~ C_{i_j^t,g} = P(g | i_j^t)

            Generate activity observation:
                a_j^t ~ E_{s_j^t,a} = P(a | s_j^t)

            Generate purchase observation:
                p_j^t ~ D_{v_j^t,p} = P(p | v_j^t)

Output:
    N synthetic behavioral sequences
```

## Simulation Design

To evaluate the interpretive capacity of the proposed model, we first define transition matrices that represent hypothetical user behavior patterns.

Using the algorithm above, we generate synthetic training samples that mimic user decision journeys. The algorithm simulates user behavioral sequences by iterating through the predefined transition matrices, which creates diverse behavioral trajectories under controlled assumptions.

After generating the synthetic data, we apply the proposed 3DHMM to predict three behavioral outcomes at the final timestamp of each user's sequence. We then compare the learned transition matrices with the true predefined transition matrices used in the simulation. This comparison evaluates whether the model can recover the latent dynamics embedded in the synthetic behavioral data.

To keep the simulation computationally feasible, we use two hidden levels for each psychological state and two possible observations for each task. This simplified setup still provides enough complexity to demonstrate the interpretability of the model.

## Interpretation of Predefined Transition Matrices

We split the generated samples into training and testing sets:

- 80% of the generated samples are used as the training set.
- 20% of the generated samples are used as the testing set.
- For each sample, the first $T-1$ timestamps are used to optimize model parameters.
- The final timestamp is used to predict user behavior.

Figure `fig-sim3` compares the predicted and true initial distributions and transition probabilities recovered by the proposed 3DHMM. The results show that the model can effectively recover the predefined synthetic transition probabilities.

The main findings are:

1. The initial probabilities of user preferences are approximately uniform, indicating an equal starting point for each state at the beginning of the simulation.
2. Users show a strong tendency to remain in their current psychological states, as reflected by the high diagonal values in the transition matrices.
3. Users at extreme engagement levels may switch to the opposite level, suggesting fluidity in engagement behavior.
4. The model also captures the behavior-generation probabilities associated with different psychological states.

## Figures

### Predicted and True Transition Probabilities

The following figure group visualizes the predicted and true initial distributions and state-transition probabilities.

| Panel | Description | File |
|---|---|---|
| Figure A | Predicted and true initial distributions $\pi$ | `fig/pi_pred_true.pdf` |
| Figure B | Probabilities of $Q_{kk'}=P(i_j^t=k'\mid i_j^{t-1}=k)$ | `fig/Q.pdf` |
| Figure C | Probabilities of $F_{ku}=P(s_j^1=u\mid i_j^1=k)$ | `fig/F.pdf` |
| Figure D | Probabilities of $N_{uo}=P(v_j^1=o\mid s_j^1=u)$ | `fig/N.pdf` |
| Figure E | Probabilities of $H_{\text{Low},k'u'}=P(s_j^t=u'\mid i_j^t=k',s_j^{t-1}=\text{Low})$ | `fig/H_0.pdf` |
| Figure F | Probabilities of $H_{\text{High},k'u'}=P(s_j^t=u'\mid i_j^t=k',s_j^{t-1}=\text{High})$ | `fig/H_1.pdf` |
| Figure G | Probabilities of $M_{\text{Low},u'o'}=P(v_j^t=o'\mid s_j^t=u',v_j^{t-1}=\text{Low})$ | `fig/M_0.pdf` |
| Figure H | Probabilities of $M_{\text{High},u'o'}=P(v_j^t=o'\mid s_j^t=u',v_j^{t-1}=\text{High})$ | `fig/M_1.pdf` |


### Observation-Generating Probabilities

The following figure group visualizes the observation-generating probabilities.

| Panel | Description | File |
|---|---|---|
| Figure A | Probabilities of $C_{kg}=P(g_j^t=g\mid i_j^t=k)$ | `fig/C.pdf` |
| Figure B | Probabilities of $E_{ua}=P(a_j^t=a\mid s_j^t=u)$ | `fig/E.pdf` |
| Figure C | Probabilities of $D_{op}=P(p_j^t=p\mid v_j^t=o)$ | `fig/D.pdf` |


## Summary

The simulation study shows that the proposed 3DHMM can recover the predefined latent transition structure and observation-generating probabilities from synthetic behavioral sequences. This provides evidence that the model can capture interpretable dynamic patterns in user behavior and can be used to analyze the evolution of psychological states in the mobile gaming context.
