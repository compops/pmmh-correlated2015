import numpy   as np
import pandas
from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import normalIID_2parameters

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
np.random.seed( 87655678 );
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
pmh.nIter       = 30000;
pmh.nBurnIn     = 1000;

pmh.rvnSamples  = 1 + sm.nPart;

pmh.nProgressReport        = 5000
pmh.writeOutProgressToFile = False;

# Set initial parameters
pmh.initPar     = ( 0.50, 0.30 )

# Set Hessian
pmh.invHessian  = 1.0;
pmh.stepSize    = 0.10;

########################################################################
# Run the ideal sampler
########################################################################
#from scipy.stats import norm
#
#class idealSampler(object):
#    typeSampler='ideal'
#    def trueLoglikelihood( self, thSys ):
#        self.ll = np.sum( norm.logpdf( thSys.y,thSys.par[0], np.sqrt(thSys.par[1]**2+thSys.par[2]**2 ) ) )
#
#idsamp = idealSampler();
#idsamp.filter = idsamp.trueLoglikelihood;
#pmh.runSampler( idsamp, sys, th );
#np.mean( pmh.th[pmh.nBurnIn:pmh.nIter,:],axis=0)
#array([ 0.51748592,  0.26525038])

########################################################################
# Run the sampler
########################################################################

gridTheta    = np.arange(0.0,1.025,0.025);
gridpGlobal  = np.arange(0.0,1.025,0.025);


res = np.zeros( ( 6, len(gridTheta), len(gridpGlobal) ) )

for jj in range( len(gridpGlobal) ):
    for ii in range( len(gridTheta) ):

        # Set random seed
        np.random.seed( 87655678 + simIdx );

        pmh.sigmaU  = gridTheta[ii];
        pmh.alpha   = gridpGlobal[jj];

        pmh.runSampler( sm, sys, th );
        #pmh.writeToFile(fileOutName='results/pmh-joe2015/example1-iid/heatmap/pmh0-N'+str(sm.nPart)+'-rvTheta'+str(ii)+'-rvpGlobal'+str(jj)+'-run'+str(simIdx)+'.csv');

        res[0,ii,jj]   = gridTheta[ii];
        res[1,ii,jj]   = gridpGlobal[jj];
        res[2,ii,jj]   = np.mean( pmh.th[ pmh.nBurnIn:pmh.nIter, : ], axis=0 );
        res[3,ii,jj]   = np.mean( pmh.accept[ pmh.nBurnIn:pmh.nIter ] );
        res[4,ii,jj]   = pmh.calcSJD();
        res[5,ii,jj]   = pmh.calcIACT(maxlag=100,thhat=(0.45439155,0.0));

        print ((ii,jj,len(gridTheta),len(gridpGlobal)))

    #print(res[:,:,jj])
    fileOut = pandas.DataFrame(res[:,:,jj].transpose(),columns=("theta","alpha","th0","accept rate","sjd(mu)","iact(mu)"));
    fileOut.to_csv('results/example1-iid-heatmap-alpha' +str(jj) +'-run' + str(simIdx) + '.csv');

#subplot(3,1,1); plot( grid, res[1,:] ); xscale('log')
#subplot(3,1,2); plot( grid, res[2,:] ); xscale('log')
#subplot(3,1,3); plot( grid, res[3,:] ); xscale('log')

########################################################################
# End of file
########################################################################
