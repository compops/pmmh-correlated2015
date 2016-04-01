##############################################################################
# Minimal working example
# Parameter inference in Gaussian IID model
# using correlated psuedo-marginal Metropolis-Hastings
#
# (c) Johan Dahlin 2016 ( johan.dahlin (at) liu.se )
##############################################################################

import numpy   as np
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
th.nParInference = 1;
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
pmh.nIter                  = 10000;
pmh.nBurnIn                = 2500;

pmh.rvnSamples             = 1 + sm.nPart;

pmh.nProgressReport        = 5000
pmh.writeOutProgressToFile = False;

# Set initial parameters
pmh.initPar                = ( 0.50, 0.30 )

# Settings for th proposal
pmh.invHessian             = 1.0;
pmh.stepSize               = 0.10;

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