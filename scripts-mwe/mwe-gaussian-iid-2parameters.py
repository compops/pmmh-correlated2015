##############################################################################
# Minimal working example
# Parameter inference in Gaussian IID model
# using correlated psuedo-marginal Metropolis-Hastings
#
# (c) Johan Dahlin 2016 ( johan.dahlin (at) liu.se )
##############################################################################

import numpy            as np
import matplotlib.pylab as plt

from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import normalIID_2parameters

np.random.seed( 87655678 );

##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
pmh              = pmh_correlatedRVs.stcPMH();


##############################################################################
# Setup the system
##############################################################################
sys               = normalIID_2parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        = 0.50;
sys.par[1]        = 0.30;
sys.par[2]        = 0.10;
sys.T             = 10;
sys.xo            = 0.0;


##############################################################################
# Generate data
##############################################################################
sys.generateData();


##############################################################################
# Setup the parameters
##############################################################################
th               = normalIID_2parameters.ssm()
th.nParInference = 2;
th.copyData(sys);


##############################################################################
# Setup the IS algorithm
##############################################################################
sm.filter          = sm.SISrv;
sm.sortParticles   = False;
sm.nPart           = 10;
sm.resampFactor    = 2.0;
sm.genInitialState = True;


##############################################################################
# Setup the PMH algorithm
##############################################################################
pmh.nIter                  = 30000;
pmh.nBurnIn                = 10000;
pmh.nProgressReport        = 5000

pmh.rvnSamples             = 1 + sm.nPart;
pmh.writeOutProgressToFile = False;

# Set initial parameters
pmh.initPar                = sys.par;

# Settings for th proposal (rule-of-thumb)
pmh.invHessian             = np.matrix([[ 0.01338002, -0.00031321],
                                        [-0.00031321,  0.00943717]]);
pmh.stepSize               = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.alpha                  = 0.00

##############################################################################
# Run the correlated pmMH algorithm
##############################################################################

# Correlated random numbers
pmh.sigmaU = 0.50
pmh.runSampler( sm, sys, th );

muCPMMH = pmh.th
iactC   = pmh.calcIACT()

# Uncorrelated random numbers (standard pmMH)
pmh.sigmaU = 1.0
pmh.runSampler( sm, sys, th );

muUPMMH = pmh.th
iactU   = pmh.calcIACT()

(iactC, iactU)

##############################################################################
# Plot the comparison
##############################################################################

plt.figure(1);
plt.subplot(2,3,1); 
plt.plot(muCPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (cpmMH)");

plt.subplot(2,3,2); 
plt.hist(muCPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(2,3,3); 
plt.acorr(muCPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (cpmMH)");

plt.subplot(2,3,4); 
plt.plot(muCPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (cpmMH)");

plt.subplot(2,3,5); 
plt.hist(muCPMMH[:,1],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(2,3,6); 
plt.acorr(muCPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (cpmMH)");

plt.figure(2);
plt.subplot(2,3,1); 
plt.plot(muUPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (pmMH)");

plt.subplot(2,3,2); 
plt.hist(muUPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(2,3,3); 
plt.acorr(muUPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (pmMH)");

plt.subplot(2,3,4); 
plt.plot(muUPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (pmMH)");

plt.subplot(2,3,5); 
plt.hist(muUPMMH[:,1],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(2,3,6); 
plt.acorr(muUPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (pmMH)");

##############################################################################
# End of file
##############################################################################