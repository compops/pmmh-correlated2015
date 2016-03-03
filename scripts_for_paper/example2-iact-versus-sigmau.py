import numpy   as np
from   para    import pmh_correlatedRVs
from   models  import probit_labour

import os
simIdx = int( os.sys.argv[1] );

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
th.nPart    = 150;

##############################################################################
# Setup the PMH algorithm
##############################################################################

# General settings
pmh.nIter                  = 10000;
pmh.nBurnIn                = 1000;
pmh.rvnSamples             = th.nPart;
pmh.nProgressReport        = 1000
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

gridTheta    = np.arange( 0.05, 1.05, 0.05 )

res = np.zeros( ( 1+2*(th.nParInference+1), len(gridTheta) ) )

for ii in range( len(gridTheta) ):

    # Set random seed
    np.random.seed( 87655678 + simIdx );

    pmh.sigmaU  = gridTheta[ii];
    pmh.alpha   = 0.0;

    pmh.runSampler( th, sys, th );
    pmh.writeToFile(fileOutName='results/example2-labour-fulloutput/pmh0-sigmau'+str(ii)+'-run'+str(simIdx)+'.csv');

    res[0,ii]                                           = gridTheta[ii];
    foo                                                 = np.mean( pmh.th[ pmh.nBurnIn:pmh.nIter, : ], axis=0 );
    res[range(1,th.nParInference+1),ii]                 = foo;
    res[th.nParInference+1,ii]                          = np.mean( pmh.accept[ pmh.nBurnIn:pmh.nIter ] );
    res[th.nParInference+2,ii]                          = pmh.calcSJD();
    foo                                                 = pmh.calcIACT( maxlag=100 );
    res[range((th.nParInference+3),res.shape[0]),ii]    = foo;

    print ( ( ii, len(gridTheta) ) )

import pandas
fileOut = pandas.DataFrame( res[:,:].transpose(), columns = ("theta","th0","th1","th2","th3","th4","th5","th6","th7","accept rate","sjd","iact0","iact1","iact2","iact3","iact4","iact5","iact6","iact7") );
fileOut.to_csv('results/example2-labour/pmmh0-run' + str(simIdx) + '.csv');
