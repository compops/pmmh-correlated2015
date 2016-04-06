##############################################################################
# Minimal working example
# Parameter inference in stochastic volatility model with leverage
# using correlated psuedo-marginal Metropolis-Hastings
#
# (c) Johan Dahlin 2016 ( johan.dahlin (at) liu.se )
##############################################################################

import numpy   as np
import Quandl
from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import hwsv_4parameters

np.random.seed( 87655678 );

##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
pmh              = pmh_correlatedRVs.stcPMH();


##############################################################################
# Setup the system
##############################################################################
sys               = hwsv_4parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        = 0.00;
sys.par[1]        = 0.98;
sys.par[2]        = 0.16;
sys.par[3]        = -0.70;
sys.T             = 745;
sys.xo            = 0.0;
sys.version       = "standard"


##############################################################################
# Generate data
##############################################################################
d = Quandl.get("NASDAQOMX/OMXS30", trim_start="2011-01-02", trim_end="2014-01-02")
y = 100 * np.diff(np.log(d['Index Value']))
y = y[~np.isnan(y)]
sys.generateData()
sys.y = y


##############################################################################
# Setup the parameters
##############################################################################
th               = hwsv_4parameters.ssm()
th.nParInference = 4;
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
pmh.nBurnIn                 = 1000;
pmh.nProgressReport         = 1000;

pmh.rvnSamples              = 1 + sm.nPart;
pmh.writeOutProgressToFile  = False;

# Settings for th proposal
pmh.initPar        = ( 0.22687995,  0.9756004 ,  0.18124849, -0.71862631 );
pmh.invHessian     = np.matrix([[  3.84374302e-02,   2.91796833e-04,  -5.30385701e-04,  -1.63398216e-03],
                                [  2.91796833e-04,   9.94254177e-05,  -2.60256138e-04,  -1.73977480e-04],
                                [ -5.30385701e-04,  -2.60256138e-04,   1.19067965e-03,   2.80879579e-04],
                                [ -1.63398216e-03,  -1.73977480e-04,   2.80879579e-04,   6.45765006e-03]])
pmh.stepSize       = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.alpha                  = 0.00

##############################################################################
# Run the correlated pmMH algorithm
##############################################################################

# Correlated random numbers
pmh.sigmaU = 0.55
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
plt.subplot(4,3,1); 
plt.plot(muCPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (cpmMH)");

plt.subplot(4,3,2); 
plt.hist(muCPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(4,3,3); 
plt.acorr(muCPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.00,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (cpmMH)");

plt.subplot(4,3,4); 
plt.plot(muCPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("phi (cpmMH)");

plt.subplot(4,3,5); 
plt.hist(muCPMMH[:,1],normed=True); 
plt.xlabel("phi"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(4,3,6); 
plt.acorr(muCPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of phi (cpmMH)");

plt.subplot(4,3,7); 
plt.plot(muCPMMH[:,2]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (cpmMH)");

plt.subplot(4,3,8); 
plt.hist(muCPMMH[:,2],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(4,3,9); 
plt.acorr(muCPMMH[:,2],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (cpmMH)");

plt.subplot(4,3,10); 
plt.plot(muCPMMH[:,3]); 
plt.xlabel("iteration"); 
plt.ylabel("rho (cpmMH)");

plt.subplot(4,3,11); 
plt.hist(muCPMMH[:,3],normed=True); 
plt.xlabel("rho"); 
plt.ylabel("posterior estimate (cpmMH)");

plt.subplot(4,3,12); 
plt.acorr(muCPMMH[:,3],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of rho (cpmMH)");


plt.figure(2);
plt.subplot(4,3,1); 
plt.plot(muUPMMH[:,0]); 
plt.xlabel("iteration"); 
plt.ylabel("mu (pmMH)");

plt.subplot(4,3,2); 
plt.hist(muUPMMH[:,0],normed=True); 
plt.xlabel("mu"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(4,3,3); 
plt.acorr(muUPMMH[:,0],maxlags=100); 
plt.axis((0,100,0.00,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of mu (pmMH)");

plt.subplot(4,3,4); 
plt.plot(muUPMMH[:,1]); 
plt.xlabel("iteration"); 
plt.ylabel("phi (pmMH)");

plt.subplot(4,3,5); 
plt.hist(muUPMMH[:,1],normed=True); 
plt.xlabel("phi"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(4,3,6); 
plt.acorr(muUPMMH[:,1],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of phi (pmMH)");

plt.subplot(4,3,7); 
plt.plot(muUPMMH[:,2]); 
plt.xlabel("iteration"); 
plt.ylabel("sigmav (pmMH)");

plt.subplot(4,3,8); 
plt.hist(muUPMMH[:,2],normed=True); 
plt.xlabel("sigmav"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(4,3,9); 
plt.acorr(muUPMMH[:,2],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of sigmav (pmMH)");

plt.subplot(4,3,10); 
plt.plot(muUPMMH[:,3]); 
plt.xlabel("iteration"); 
plt.ylabel("rho (cpmMH)");

plt.subplot(4,3,11); 
plt.hist(muUPMMH[:,3],normed=True); 
plt.xlabel("rho"); 
plt.ylabel("posterior estimate (pmMH)");

plt.subplot(4,3,12); 
plt.acorr(muUPMMH[:,3],maxlags=100); 
plt.axis((0,100,0.95,1))
plt.xlabel("iteration"); 
plt.ylabel("acf of rho (pmMH)");

##############################################################################
# End of file
##############################################################################