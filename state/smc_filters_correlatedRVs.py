##############################################################################
##############################################################################
# Routines for
# Particle filtering with fixed random numbers
#
# Copyright (c) 2016 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from scipy.stats             import norm
from smc_resampling          import *
from smc_helpers             import *

##########################################################################
# Particle filtering with fixed random variables
##########################################################################

def proto_rvpf(classSMC,sys):

    # Set the filter type and save T
    classSMC.T = sys.T;

    # Check algorithm settings and set to default if needed
    setSettings(classSMC,"filter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    ar  = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    pt  = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ll  = np.zeros(sys.T);
    ess = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialStateRV( classSMC.nPart, classSMC.rv[:,0] );
        w[:,0] = 1.0 / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;
        w[:,0] = 1.0 / classSMC.nPart;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(1, sys.T):

        #=============================================================
        # Resample particles
        #=============================================================

        # If resampling is enabled
        if ( classSMC.resamplingInternal == 1 ):

            # Calculate ESS
            ess[tt] = ( np.sum( w[:,tt-1]**2 ) )**(-1)

            # Check if ESS if below threshold, then resample
            if ( ess[tt] < ( classSMC.nPart * classSMC.resampFactor )  ):

                if classSMC.resamplingType == "stratified":
                    nIdx            = resampleStratified( w[:,tt-1], u = norm.cdf( classSMC.rv[0,tt] ) );
                    nIdx            = np.transpose(nIdx.astype(int));
                    pt[:,tt]        = p[nIdx,tt-1];
                    ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                    ar[:,tt]        = nIdx;
                    a[:,tt]         = nIdx;
                elif classSMC.resamplingType == "systematic":
                    nIdx            = resampleSystematic( w[:,tt-1], u = norm.cdf( classSMC.rv[0,tt] ) );
                    nIdx            = np.transpose(nIdx.astype(int));
                    pt[:,tt]        = p[nIdx,tt-1];
                    ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                    ar[:,tt]        = nIdx;
                    a[:,tt]         = nIdx;
                elif classSMC.resamplingType == "multinomial":
                    nIdx            = resampleMultinomial( w[:,tt-1], u = norm.cdf( classSMC.rv[0,tt] ) );
                    nIdx            = np.transpose(nIdx.astype(int));
                    pt[:,tt]        = p[nIdx,tt-1];
                    ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                    ar[:,tt]        = nIdx;
                    a[:,tt]         = nIdx;
            else:
                # No resampling
                nIdx                = np.arange(0,classSMC.nPart);
                nIdx                = np.transpose(nIdx.astype(int));
                pt[:,tt]            = p[nIdx,tt-1];
                a[:,tt]             = nIdx;
                ar[:,tt]            = nIdx;

        else:
            pt[:,tt]            = p[:,tt-1];

        #=============================================================
        # Propagate particles
        #=============================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            p[:,tt] = sys.generateStateRV   ( pt[:,tt], tt-1, classSMC.rv[:,tt] );
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1))):
            p[:,tt] = sys.generateStateFARV ( pt[:,tt], tt-1, classSMC.rv[:,tt] );

        if ( ( classSMC.sortParticles == True ) & ( classSMC.resamplingInternal == 1 ) ):
            p[:,tt] = np.sort( p[:,tt] );

        #=================================================================
        # Weight particles
        #=================================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt );
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            w[:,tt] = sys.evaluateObservationFA ( p[:,tt], tt );

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt]   = wmax + np.log(np.sum(w[:,tt])) - np.log(classSMC.nPart);
        w[:,tt] /= np.sum(w[:,tt]);

        # Calculate the normalised filter weights (1/N) as it is a FAPF
        if ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            v[:,tt] = w[:,tt];
            w[:,tt] = np.ones(classSMC.nPart) / classSMC.nPart;

        # Estimate the filtered state
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================

    # Sample a trajectory
    idx            = np.random.choice( classSMC.nPart, 1, p=w[:,sys.T-1] )
    idx            = ar[idx,sys.T-1].astype(int);
    classSMC.xtraj = p[idx,:]

    # Compile the rest of the output
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum( ll );
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.a     = a;
    classSMC.ar    = ar;
    classSMC.p     = p;
    classSMC.pt    = pt;
    classSMC.ess   = ess;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################