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
sys.loadData('data/pmh_joe2015/mroz.csv');

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

##############################################################################
# Check correlation in the likelihood estimator
##############################################################################

nIter           = 2500;
ll0             = np.zeros( nIter )
ll1             = np.zeros( nIter )

pmh.rvpGlobal   = 0.0;
sigmauGrid      = np.arange( 0.00,1.05,0.05 );
nPartGrid       = ( 1, 2, 5, 10, 20, 50, 100 )

covPMH          = np.zeros( ( len(sigmauGrid), len(nPartGrid), 3 ) )

for ii in range(len(sigmauGrid)):

    pmh.sigmaU = sigmauGrid[ii];
    pmh.alpha  = 0.0;

    for jj in range( len(nPartGrid) ):

        th.nPart        = nPartGrid[jj];
        pmh.rvnSamples  = 1 + th.nPart;

        for kk in range(nIter):

            # Sample initial random variables and compute likelihood estimate
            pmh.rv          = np.random.normal( size=( pmh.rvnSamples, sys.T ) );
            th.rv           = pmh.rv;

            th.filter( th );
            ll0[ kk ]        = th.ll;

            # Propose new random variables ( Local move )
            u = np.random.uniform()
            pmh.rvp    = np.sqrt( 1.0 - pmh.sigmaU**2 ) * pmh.rv + pmh.sigmaU * np.random.normal( size=(pmh.rvnSamples,sys.T) );

            # Compute likelihood estimate
            th.rv     = pmh.rvp;
            th.filter( th );
            ll1[ kk ] = th.ll;

        covPMH[ii,jj,0] = np.var( ll0 )
        covPMH[ii,jj,1] = np.cov( ll0, ll1 )[0,1]
        covPMH[ii,jj,2] = np.corrcoef( ll0, ll1 )[0,1]

        print( (ii,len(sigmauGrid),jj,len(nPartGrid) ) );



figure(1)
for jj in range( len(nPartGrid) ):
    plot(sigmauGrid,covPMH[:,jj,2])

#
import pandas
fileOut = pandas.DataFrame( covPMH[:,:,0],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example2-correlation-versus-sigmau-llvar.csv');
fileOut = pandas.DataFrame( covPMH[:,:,1],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example2-correlation-versus-sigmau-llcov.csv');
fileOut = pandas.DataFrame( covPMH[:,:,2],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example2-correlation-versus-sigmau-llcorr.csv');

#sqrt( var( ll0 ) )

########################################################################
# End of file
########################################################################
