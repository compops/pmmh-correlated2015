import numpy   as np
from   para    import pmh_correlatedRVs
from   models  import probit_labour

##############################################################################
# Arrange the data structures
##############################################################################
pmh              = pmh_correlatedRVs.stcPMH();

##############################################################################
# Setup the system
##############################################################################
sys               = probit_labour.ssm()

sys.loadData('data/pmh_joe2015/mroz.csv')

##############################################################################
# Setup the parameters
##############################################################################
th               = probit_labour.ssm()
th.nParInference = sys.nRegressors + 1;
th.copyData(sys);

##############################################################################
# Setup the importance sampler
##############################################################################
th.filter   = th.estimateLogLikelihood;
th.xtraj    = np.zeros( sys.T );
th.nPart    = 250;

##############################################################################
# Setup the PMH algorithm
##############################################################################

# General settings
pmh.nIter                  = 20000;
pmh.nBurnIn                = 5000;
pmh.rvnSamples             = th.nPart;
pmh.nProgressReport        = 5000
pmh.writeOutProgressToFile = False;

# Set initial parameters and Hessian (only used for random walk)
#pmh.initPar     = sys.par;
#pmh.invHessian  = np.diag( ( np.array( ( 0.1326, 0.0058, 0.0109, 0.0108, 0.0005, 0.0031, 0.2317, 0.0703 ) )**2) [ range(th.nParInference) ] );

pmh.initPar     = np.array([ 0.15115544, -0.01172118,  0.13007564,  0.12224231, -0.00184684, -0.05017923, -0.84091536,  0.04165484])
pmh.invHessian  = np.loadtxt("data/pmh_joe2015/mroz-rwwalk-cov.csv");
pmh.invHessian  = pmh.invHessian[ 0:th.nParInference, 0:th.nParInference ]
pmh.stepSize    = 2.562 / np.sqrt(th.nParInference);


########################################################################
# Run the sampler
########################################################################

# Set random seed
np.random.seed( 87655678 );

pmh.sigmaU  = 0.4;
pmh.alpha   = 0.0;

pmh.runSampler( th, sys, th );
#pmh.writeToFile(fileOutName='results/example2-rwproposal-sigmau1.csv');

#res = cov( (pmh.th).transpose() )
#np.savetxt("data/pmh_joe2015/mroz-rwwalk-cov.csv",res);

res1 = np.mean( pmh.th[ pmh.nBurnIn:pmh.nIter, : ], axis= 0 )
res2 = np.sqrt( np.var( pmh.th[ pmh.nBurnIn:pmh.nIter, : ], axis= 0 ) )

np.vstack((res1,res2)).transpose()


figure(1);

for ii in range( th.nParInference ):
    subplot(4,2,ii+1);
    hist( pmh.th[ pmh.nBurnIn:pmh.nIter, ii ], bins=np.floor(np.sqrt(pmh.nIter-pmh.nBurnIn)) )
