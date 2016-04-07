# pmmh-correlated2015
Accelerating pseudo-marginal Metropolis-Hastings by correlating auxiliary variables

## Dependencies

## Minimal working examples (scripts-mwe)

### mwe-gaussian-iid
The first minimal working example is parameter inference in a Gaussian IID model (see section 4.1 in the paper). That is, IID data generated from a Gaussian distribution with some mean and standard deviation. The mean parameter is estimated in **mwe-gaussian-iid-1parameter.py** and both the mean and standard deviation in **mwe-gaussian-iid-2parameters.py**. The script is probably quite self-explained. The proposal for theta (when both parameters are estimated) is based on the rule-of-thumb proposed by Sherlock, Thiery, Robert and Rosenthal (2015) 
On the efficiency of pseudo-marginal random walk Metropolis algorithms avaiable from < http://arxiv.org/pdf/1309.7209 > using a pilot run. 

We are mainly concerned with comparing the IACT computed from the autocorrelation function. For the case when only the mean is estimated, we calibrate the model by

``` python
pmh.invHessian  = 1.0;
pmh.stepSize    = 0.1;
pmh.sigmaU      = 0.50
```

This means that the standard deviation of the Gaussian random walk proposal for the parameter is 0.1. The standard deviation for the CN proposal for u is 0.50. The resulting IACTs for using correlated random variables (sigmaU=0.50) and standard pmMH with uncorrelated random variables (sigmaU=1.0) are:

``` python
>>> (iactC, iactU)
(array([ 46.558]), array([ 48.762]))
```
where we see that the IACT decreases from 48.762 to 46.558 when introducing correlation in the random variables. This corresponds to a modest increase in efficency by using correlated random numbers. We interprete these values as 49 iterations are required for the uncorrelated algorithm to provide an independent sample compared with 47 for the correlated algorithm.

The change is larger when we estimate both parameters in the model. Here, we calibrate the model by

``` python
# Settings for th proposal (rule-of-thumb)
pmh.invHessian = np.matrix([[ 0.01338002, -0.00031321],
                            [-0.00031321,  0.00943717]]);
pmh.stepSize   = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.alpha      = 0.00
pmh.sigmaU     = 0.50
```
where we make use of the aforementioned rule-of-thumb to calibrate the parameter proposal. In this case, we obtain the following IACT:

``` python
>>> (iactC, iactU)
(array([ 50.652,  64.194]), array([  68.696,  102.478]))
```
where we see that the maximum IACT decreases from 102.478 to 64.194 when introducing correlation in the random variables. This corresponds to a decrease by about 45% in the IACT, which corresponds to a similar possible reduction in the computational cost while keeping the same accuracy.

### mwe-sv-with-leverage
The second minimal working example is parameter inference in a stochastic volatility model with leverage using real-world data (see section 4.2 in the paper). The data is log-returns from the years 2011 to 2013 from the NASDAQ OMXS30 index, i.e. the 30 most traded stocks on the Stockholm stock exchange. The model is presented in the paper and we would like to infer all four parameters. We make use of the rule-of-thumb and pilot runs to determine the following settings:

``` python
pmh.initPar    = ( 0.22687995,  0.9756004 ,  0.18124849, -0.71862631 );
pmh.invHessian = np.matrix([[  3.84374302e-02,   2.91796833e-04,  -5.30385701e-04,  -1.63398216e-03],
                             [  2.91796833e-04,   9.94254177e-05,  -2.60256138e-04,  -1.73977480e-04],
                             [ -5.30385701e-04,  -2.60256138e-04,   1.19067965e-03,   2.80879579e-04],
                             [ -1.63398216e-03,  -1.73977480e-04,   2.80879579e-04,   6.45765006e-03]])
pmh.stepSize   = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.alpha      = 0.00
pmh.sigmaU     = 0.55
```
The resulting IACTs for the four parameters using sigmaU=0.55 and sigmaU=1.0 (standard pmMH with uncorrelated random variables) are:

``` python
>>> (iactC, iactU)
(array([ 29.272,  21.692,  21.674,  28.494]), array([ 44.512,  53.336,  52.948,  57.326]))
```
where we see that the maximum IACT decreases from 57.236 to 29.62 when introducing correlation in the random variables. This corresponds to a decrease by about 50% in the IACT, which corresponds to a similar possible reduction in the computational cost while keeping the same accuracy.

## Replication scripts for paper (scripts-draft1)

## Modification of inference in other models
