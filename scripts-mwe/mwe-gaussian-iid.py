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
sm.nPart           = 50;
sm.resampFactor    = 2.0;
sm.genInitialState = True;


##############################################################################
# Setup the PMH algorithm
##############################################################################
pmh.nIter                  = 10000;
pmh.nBurnIn                = 2500;
pmh.nProgressReport        = 1000

pmh.rvnSamples             = 1 + sm.nPart;
pmh.writeOutProgressToFile = False;

# Set initial parameters
pmh.initPar                = sys.par;

# Settings for th proposal
pmh.invHessian             = np.matrix([[ 0.01338002, -0.00031321],
                                        [-0.00031321,  0.00943717]]);
pmh.stepSize               = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal

pmh.alpha                  = 0.00


##############################################################################
# Run the correlated pmMH algorithm and plot the results
##############################################################################

pmh.sigmaU = 0.35
pmh.runSampler( sm, sys, th );

plt.figure(1);
plt.subplot(2,3,1); 
plt.plot(pmh.th[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu");

plt.subplot(2,3,2); 
plt.hist(pmh.th[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate");

plt.subplot(2,3,3); 
plt.acorr(pmh.th[:,0],maxlags=500); 
plt.axis((0,500,0.85,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu");

plt.subplot(2,3,4); 
plt.plot(pmh.th[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav");

plt.subplot(2,3,5); 
plt.hist(pmh.th[:,1],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate");

plt.subplot(2,3,6); 
plt.acorr(pmh.th[:,1],maxlags=500); 
plt.axis((0,500,0.85,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav");

##############################################################################
# Run the standard pmMH algorithm and plot the results
##############################################################################

pmh.sigmaU = 1.0
pmh.runSampler( sm, sys, th );

plt.figure(2);
plt.subplot(2,3,1); 
plt.plot(pmh.th[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu");

plt.subplot(2,3,2); 
plt.hist(pmh.th[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate");

plt.subplot(2,3,3); 
plt.acorr(pmh.th[:,0],maxlags=500); 
plt.axis((0,500,0.85,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu");

plt.subplot(2,3,4); 
plt.plot(pmh.th[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav");

plt.subplot(2,3,5); 
plt.hist(pmh.th[:,1],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate");

plt.subplot(2,3,6); 
plt.acorr(pmh.th[:,1],maxlags=500); 
plt.axis((0,500,0.85,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav");

##############################################################################
# End of file
##############################################################################