# pmmh-correlated2015
This code was downloaded from < https://github.com/compops/pmmh-correlated2015 > and contains the code used to produce the results in

J. Dahlin, F. Lindsten, J. Kronander and T. B. Schön, **Accelerating pseudo-marginal Metropolis-Hastings by correlating auxiliary variables**. Pre-print, arXiv:1512:05483v1, 2015.

The paper is available as a preprint from < http://arxiv.org/pdf/1512.05483 >.

## Dependencies
The code is written and tested for Python 2.7.6. The implementation makes use of NumPy 1.9.2, SciPy 0.15.1, Matplotlib 1.4.3, Pandas 0.13.1 and Quandl 2.8.9. On Ubuntu, these packages can be installed/upgraded using 
``` bash
sudo pip install --upgrade package-name
```
For more information about the Quandl library, see < https://www.quandl.com/tools/python >.

## Minimal working examples (scripts-mwe)
These are three minimal working examples to present how to make use of the correlated pmMH algorithm. Below, we discuss how to calibrate the algorithm and compare it with the uncorrelated version of the algorithm.

### mwe-gaussian-iid
The first minimal working example is parameter inference in a Gaussian IID model (see section 4.1 in the paper). That is, IID data generated from a Gaussian distribution with some mean and standard deviation. The mean parameter is estimated in **mwe-gaussian-iid-1parameter.py** and both the mean and standard deviation in **mwe-gaussian-iid-2parameters.py**. The script is probably quite self-explained. The proposal for theta (when both parameters are estimated) is based on the rule-of-thumb proposed by Sherlock, Thiery, Robert and Rosenthal (2015) 
On the efficiency of pseudo-marginal random walk Metropolis algorithms available from < http://arxiv.org/pdf/1309.7209 > using a pilot run. 

We are mainly concerned with comparing the IACT computed from the autocorrelation function. For the case when only the mean is estimated, we calibrate the model by

``` python
pmh.invHessian = 1.0
pmh.stepSize = 0.1
pmh.sigmaU = 0.50
```

This means that the standard deviation of the Gaussian random walk proposal for the parameter is 0.1. The standard deviation for the CN proposal for u is 0.50. The resulting IACTs for using correlated random variables (sigmaU=0.50) and standard pmMH with uncorrelated random variables (sigmaU=1.0) are:

``` python
>>> (iactC, iactU)
(array([46.558]), array([48.762]))
```
where we see that the IACT decreases from 48.762 to 46.558 when introducing correlation in the random variables. This corresponds to a modest increase in efficiency by using correlated random numbers. We interpret these values as 49 iterations are required for the uncorrelated algorithm to provide an independent sample compared with 47 for the correlated algorithm.

The change is larger when we estimate both parameters in the model. Here, we calibrate the model by

``` python
# Settings for th proposal (rule-of-thumb)
pmh.invHessian = np.matrix([[ 0.01338002, -0.00031321],
                            [-0.00031321,  0.00943717]])
pmh.stepSize = 2.562 / np.sqrt(th.nParInference)

# Settings for u proposal
pmh.alpha = 0.00
pmh.sigmaU = 0.50
```
where we make use of the aforementioned rule-of-thumb to calibrate the parameter proposal. In this case, we obtain the following IACT:

``` python
>>> (iactC, iactU)
(array([50.652,  64.194]), array([68.696,  102.478]))
```
where we see that the maximum IACT decreases from 102.478 to 64.194 when introducing correlation in the random variables. This corresponds to a decrease by about 45% in the IACT, which corresponds to a similar possible reduction in the computational cost while keeping the same accuracy.

### mwe-sv-with-leverage
The second minimal working example is parameter inference in a stochastic volatility model with leverage using real-world data (see section 4.2 in the paper). The data is log-returns from the years 2011 to 2013 from the NASDAQ OMXS30 index, i.e. the 30 most traded stocks on the Stockholm stock exchange. The model is presented in the paper and we would like to infer all four parameters. We make use of the rule-of-thumb and pilot runs to determine the following settings:

``` python
pmh.initPar = (0.22687995,  0.9756004 ,  0.18124849, -0.71862631)
pmh.invHessian = np.matrix([[  3.84374302e-02,   2.91796833e-04,  -5.30385701e-04,  -1.63398216e-03],
                             [  2.91796833e-04,   9.94254177e-05,  -2.60256138e-04,  -1.73977480e-04],
                             [ -5.30385701e-04,  -2.60256138e-04,   1.19067965e-03,   2.80879579e-04],
                             [ -1.63398216e-03,  -1.73977480e-04,   2.80879579e-04,   6.45765006e-03]])
pmh.stepSize = 2.562 / np.sqrt(th.nParInference)

# Settings for u proposal
pmh.alpha = 0.00
pmh.sigmaU = 0.55
```
The resulting IACTs for the four parameters using sigmaU=0.55 and sigmaU=1.0 (standard pmMH with uncorrelated random variables) are:

``` python
>>> (iactC, iactU)
(array([29.272,  21.692,  21.674,  28.494]), array([44.512,  53.336,  52.948,  57.326]))
```
where we see that the maximum IACT decreases from 57.236 to 29.62 when introducing correlation in the random variables. This corresponds to a decrease by about 50% in the IACT, which corresponds to a similar possible reduction in the computational cost while keeping the same accuracy.

## Replication scripts for paper (scripts-draft1)
These scripts replicate the two examples in the paper: (1) a Gaussian IID model and (2) a stochastic volatility (SV) model with leverage. Some of the scripts needs to be executed on a cluster or similar as it requires many parallel runs of the algorithms with different parameters (sigmaU and alpha). The details are discussed below.

### exampleN-correlation-versus-sigmau.py
These two scripts replicate Figure 2 for the Gaussian IID model and creates a similar plot for the SV model. That is, computes the correlated in two consecutive log-likelihood estimates (keeping theta fixed) for different correlations in the random variables. A small variation is that this is done for a varying number of particles (N). The plots are recreated by executing the file and the output is also saved as csv-files.

### exampleN-iact-versus-sigmau(-alpha).py
These two scripts compute estimates of the IACT over a grid of different values of sigmaU and alpha. The output is the estimate of the mean of the parameter posterior, the squared jump distance and the IACT. All these variables are written to a file. In the paper, this is replicated for 32 independent Monte Carlo runs and the input variable simIdx is varied between 0 and 31 in the paper.

### theory-optimal-sigmau-for-CN
This script recreates Figure 1 by a Peskun analysis. The code is a Python implementation of the C-code provided by Yang and Rodríguez (2013) < http://www.pnas.org/content/110/48/19307.abstract > from their homepage < http://abacus.gene.ucl.ac.uk/software/MCMCjump.html >. 

## Modification of inference in other models
This code is fairly general and can be used for inference in any state space model expressed by densities and with a scalar state. The main alteration required for using multivariate states is to rewrite the particle vector, propagation step and weighting step in the particle filter.

The models are defined by files in models/. To implement a new model you can alter the existing models and re-define the functions `generateInitialStateRV`,`generateStateRV`, `evaluateObservation`. These functions take `nPart`, `xtt`, `xt`, `yt`, `tt` and `u` as inputs. `nPart` and `u` denote the number of particles and the random variables used in the particle filter. `xtt`, `xt` and `yt` denotes the next state, the current state and the current observation. Finally, `tt` denotes the current time steps. The parameters of the model are available as `self.par` and can be used for proposing and weighting particles. Finally, note that `generateState` and `generateObservation` need to be modified if you generate data from the model. Please, let me know if you need any help with this and I will try my best to sort it out.
