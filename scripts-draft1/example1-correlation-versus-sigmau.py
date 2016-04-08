##############################################################################
##############################################################################
#
# Replicates Figure 2 in
#
# J. Dahlin, F. Lindsten, J. Kronander and T. B. Schön, 
# Accelerating pmMH by correlating auxiliary variables. 
# Pre-print, arXiv:1512:05483v1, 2015.
#
# Copyright (c) 2016 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy   as np
from   state   import smc
from   para    import pmh_correlatedRVs
from   models  import normalIID_2parameters


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
# Setup the SMC algorithm
##############################################################################
sm.filter          = sm.SISrv;
sm.sortParticles   = False;
sm.nPart           = 10;
sm.resampFactor    = 2.0;
sm.genInitialState = True;


##############################################################################
# Check correlation in the likelihood estimator
##############################################################################

nIter           = 2500;
ll0             = np.zeros( nIter )
ll1             = np.zeros( nIter )

pmh.rvpGlobal   = 0.0;
sigmauGrid      = np.arange( 0.00,1.05,0.05 );
nPartGrid       = ( 1, 2, 5, 10, 20, 50 )

covPMH          = np.zeros( ( len(sigmauGrid), len(nPartGrid), 3 ) )

for ii in range(len(sigmauGrid)):

    pmh.sigmaU = sigmauGrid[ii];
    pmh.alpha  = 0.0;

    for jj in range( len(nPartGrid) ):

        sm.nPart        = nPartGrid[jj];
        pmh.rvnSamples  = 1 + sm.nPart;

        for kk in range(nIter):

            # Sample initial random variables and compute likelihood estimate
            pmh.rv          = np.random.normal( size=( pmh.rvnSamples, sys.T ) );
            sm.rv           = pmh.rv;

            sm.filter( th );
            ll0[ kk ]        = sm.ll;

            # Propose new random variables ( Local move )
            u = np.random.uniform()
            pmh.rvp    = np.sqrt( 1.0 - pmh.sigmaU**2 ) * pmh.rv + pmh.sigmaU * np.random.normal( size=(pmh.rvnSamples,sys.T) );

            # Compute likelihood estimate
            sm.rv     = pmh.rvp;
            sm.filter( th );
            ll1[ kk ] = sm.ll;

        covPMH[ii,jj,0] = np.var( ll0 )
        covPMH[ii,jj,1] = np.cov( ll0, ll1 )[0,1]
        covPMH[ii,jj,2] = np.corrcoef( ll0, ll1 )[0,1]

        print( (ii,len(sigmauGrid),jj,len(nPartGrid) ) );


from scipy.stats import linregress


figure(1)
for jj in range( len(nPartGrid) ):
    plot(sigmauGrid,covPMH[:,jj,2])



import pandas
fileOut = pandas.DataFrame( covPMH[:,:,0],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example1-correlation-versus-sigmau-llvar.csv');
fileOut = pandas.DataFrame( covPMH[:,:,1],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example1-correlation-versus-sigmau-llcov.csv');
fileOut = pandas.DataFrame( covPMH[:,:,2],index=sigmauGrid, columns=nPartGrid);
fileOut.to_csv('example1-correlation-versus-sigmau-llcorr.csv');

#sqrt( var( ll0 ) )

########################################################################
# End of file
########################################################################
