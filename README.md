# pmmh-correlated2015
Accelerating pseudo-marginal Metropolis-Hastings by correlating auxiliary variables

## mwe-gaussian-iid
The first two minimal working examples are parameter inference in a Gaussian IID model. That is, IID data generated from a Gaussian distribution with some mean and standard deviation. The mean parameter is estimated in **mwe-gaussian-iid-1parameter.py** and both the mean and standard deviation in **mwe-gaussian-iid-2parameters.py**. The script is probably quite self-explained. The proposal for theta (when both parameters are estimated) is based on the rule-of-thumb proposed by Sherlock, Thiery, Robert and Rosenthal (2015) 
On the efficiency of pseudo-marginal random walk Metropolis algorithms avaiable from < http://arxiv.org/pdf/1309.7209 > using a pilot run. 

We are mainly concerned with comparing the IACT computed from the autocorrelation function. For the case when one parameter is estimated, we have the IACT:
 
``` python
>>> (iactC, iactU)
(array([ 46.558]), array([ 48.762]))
```

which means that we have a modest increase in efficency by using correlated random numbers. We interprete these values as 49 iterations are required for the uncorrelated algorithm to provide an independent sample compared with 47 for the correlated algorithm.

The change is larger when we estimate both parameters in the model. In this case, we obtain the following IACT:

``` python
>>> (iactC, iactU)
(array([ 50.652,  64.194]), array([  68.696,  102.478]))
```

which means that we decrease the autocorrelation in the Markov chain by about 40% when introducing correlation in the random numbers. 

## mwe-lgss


## mwe-sv-with-leverage
