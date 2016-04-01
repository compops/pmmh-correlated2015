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
sys.par[0]        = 0.50;
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
sm.nPart           = 100;
sm.resampFactor    = 2.0;
sm.genInitialState = True;


##############################################################################
# Setup the PMH algorithm
##############################################################################
pmh.nIter                   = 10000;
pmh.nBurnIn                 = 1000;
pmh.nProgressReport         = 5000;

pmh.rvnSamples              = 1 + sm.nPart;
pmh.writeOutProgressToFile  = False;

# Settings for th proposal
pmh.initPar        = sys.th;
pmh.invHessian     = np.matrix([[  3.84374302e-02,   2.91796833e-04,  -5.30385701e-04,  -1.63398216e-03],
                                [  2.91796833e-04,   9.94254177e-05,  -2.60256138e-04,  -1.73977480e-04],
                                [ -5.30385701e-04,  -2.60256138e-04,   1.19067965e-03,   2.80879579e-04],
                                [ -1.63398216e-03,  -1.73977480e-04,   2.80879579e-04,   6.45765006e-03]])
pmh.stepSize       = 2.562 / np.sqrt(th.nParInference);

# Settings for u proposal
pmh.sigmaU                 = 0.50
pmh.alpha                  = 0.00


##############################################################################
# Run the sampler
##############################################################################

pmh.runSampler( sm, sys, th );


##############################################################################
# End of file
##############################################################################