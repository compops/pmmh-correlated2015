import numpy   as np
import pandas
from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import hwsv_4parameters

import os
simIdx = int( os.sys.argv[1] );

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
sys.T             = 747;
sys.xo            = 0.0;
sys.version       = "standard"


##############################################################################
# Generate data
##############################################################################
sys.generateData(fileName="data/pmh_joe2015/omxs30_20110110_20140101.csv",order="y");


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
pmh.nProgressReport         = 5000;

pmh.rvnSamples              = 1 + sm.nPart;
pmh.writeOutProgressToFile  = False;


# Set initial parameters and the settings for the proposal
pmh.initPar        = ( 0.22687995,  0.9756004 ,  0.18124849, -0.71862631 );
pmh.invHessian     = np.matrix([[  3.84374302e-02,   2.91796833e-04,  -5.30385701e-04,  -1.63398216e-03],
                                [  2.91796833e-04,   9.94254177e-05,  -2.60256138e-04,  -1.73977480e-04],
                                [ -5.30385701e-04,  -2.60256138e-04,   1.19067965e-03,   2.80879579e-04],
                                [ -1.63398216e-03,  -1.73977480e-04,   2.80879579e-04,   6.45765006e-03]])
pmh.stepSize       = 2.562 / np.sqrt(th.nParInference);

# Estimate standard deviation in the log-likelihood
#th.storeParameters( pmh.initPar, sys );
#llgrid = np.zeros( 500 );
#
#for kk in range( 500 ):
#    sm.rv = np.random.normal( size=(pmh.rvnSamples,sys.T ) )
#    sm.filter( th );
#    llgrid[kk] = sm.ll;
#    print(kk)

########################################################################
# Run the sampler
########################################################################

gridTheta    = np.arange( 0.05, 1.05, 0.05 )

res = np.zeros( ( 11, len(gridTheta) ) )

for ii in range( len(gridTheta) ):

    # Set random seed
    np.random.seed( 87655678 + simIdx );

    pmh.sigmaU     = gridTheta[ii];
    pmh.alpha      = 0.0;

    pmh.runSampler( sm, sys, th );
    pmh.writeToFile(fileOutName='results/example3-OMXS30-fulloutput/pmh0-sigmau'+str(ii)+'-run'+str(simIdx)+'.csv');

    res[0,ii]   = gridTheta[ii];
    foo         = np.mean( pmh.th[ pmh.nBurnIn:pmh.nIter, : ], axis=0 );
    res[1,ii]   = foo[0];
    res[2,ii]   = foo[1];
    res[3,ii]   = foo[2];
    res[4,ii]   = foo[3];
    res[5,ii]   = np.mean( pmh.accept[ pmh.nBurnIn:pmh.nIter ] );
    res[6,ii]   = pmh.calcSJD();
    foo         = pmh.calcIACT( maxlag=100 );
    res[7,ii]   = foo[0];
    res[8,ii]   = foo[1];
    res[9,ii]   = foo[2];
    res[10,ii]  = foo[3];

    print ( ( ii, len(gridTheta) ) )

fileOut = pandas.DataFrame( res[:,:].transpose(), columns = ("theta","th0","th1","th2","th3","accept rate","sjd","iact0","iact1","iact2","iact3") );
fileOut.to_csv('results/example3-OMXS30/pmmh0-run' + str(simIdx) + '.csv');


########################################################################
# End of file
########################################################################
