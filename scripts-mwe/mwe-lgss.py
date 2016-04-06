##############################################################################
# Minimal working example
# Parameter inference in linear Gaussian state space model
# using correlated psuedo-marginal Metropolis-Hastings
#
# (c) Johan Dahlin 2016 ( johan.dahlin (at) liu.se )
##############################################################################

import numpy   as np
from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import lgss_4parameters

np.random.seed( 87655678 );

##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
pmh              = pmh_correlatedRVs.stcPMH();


##############################################################################
# Setup the system
##############################################################################
sys               = lgss_4parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        = 0.20;
sys.par[1]        = 0.80;
sys.par[2]        = 1.00;
sys.par[3]        = 0.10;
sys.T             = 100;
sys.xo            = 0.0;


##############################################################################
# Generate data
##############################################################################
sys.generateData();


##############################################################################
# Setup the parameters
##############################################################################
th               = lgss_4parameters.ssm()
th.nParInference = 3;
th.copyData(sys);


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter          = sm.bPFrv;
sm.sortParticles   = True;
sm.nPart           = 50;
sm.resampFactor    = 2.0;
sm.genInitialState = True;


##############################################################################
# Setup the PMH algorithm
##############################################################################
pmh.nIter                   = 10000;
pmh.nBurnIn                 = 2500;
pmh.nProgressReport         = 1000;

pmh.rvnSamples              = 1 + sm.nPart;
pmh.writeOutProgressToFile  = False;

# Settings for th proposal
pmh.initPar        = sys.par;
pmh.invHessian     = np.array([[ 0.03708295, -0.0008457 , -0.00219988],
                               [-0.0008457 ,  0.00032924,  0.00044877],
                               [-0.00219988,  0.00044877,  0.00354022]])
       
pmh.stepSize       = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.alpha          = 0.00


##############################################################################
# Run the correlated pmMH algorithm
##############################################################################

# Correlated random numbers
pmh.sigmaU = 0.0001
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
plt.subplot(3,3,1); 
plt.plot(muCPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (cpmMH)");

plt.subplot(3,3,2); 
plt.hist(muCPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(3,3,3); 
plt.acorr(muCPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (cpmMH)");

plt.subplot(3,3,4); 
plt.plot(muCPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("phi (cpmMH)");

plt.subplot(3,3,5); 
plt.hist(muCPMMH[:,1],normed=True); 
plt.xlabel("phi"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(3,3,6); 
plt.acorr(muCPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of phi (cpmMH)");

plt.subplot(3,3,7); 
plt.plot(muCPMMH[:,2]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (cpmMH)");

plt.subplot(3,3,8); 
plt.hist(muCPMMH[:,2],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(3,3,9); 
plt.acorr(muCPMMH[:,2],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (cpmMH)");



plt.figure(2);
plt.subplot(3,3,1); 
plt.plot(muUPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (pmMH)");

plt.subplot(3,3,2); 
plt.hist(muUPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(3,3,3); 
plt.acorr(muUPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (pmMH)");

plt.subplot(3,3,4); 
plt.plot(muUPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("phi (cpmMH)");

plt.subplot(3,3,5); 
plt.hist(muUPMMH[:,1],normed=True); 
plt.xlabel("phi"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(3,3,6); 
plt.acorr(muUPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of phi (cpmMH)");

plt.subplot(3,3,7); 
plt.plot(muUPMMH[:,2]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (cpmMH)");

plt.subplot(3,3,8); 
plt.hist(muUPMMH[:,2],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(3,3,9); 
plt.acorr(muUPMMH[:,2],maxlags=100); 
plt.axis((0,100,0.90,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (cpmMH)");

##############################################################################
# End of file
##############################################################################