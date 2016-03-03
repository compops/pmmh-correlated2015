##############################################################################
##############################################################################
# Routines for
# Particle smoothing
# Version 2015-03-12
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from smc_resampling          import *
from smc_helpers             import *
from smc_additivefunctionals import *

##########################################################################
# Particle smoother: filter smooothing
##########################################################################
def proto_fsPS(classSMC,sys,fixParticles=False):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "fs"
    setSettings(classSMC,"smoother");

    # Run initial filter
    if ( fixParticles == False ):
        classSMC.filter(sys);

    # Initialise variables
    xs    = np.zeros((sys.T,1));
    g0    = np.zeros((sys.nParInference,sys.T));
    g1    = np.zeros((classSMC.nPart,sys.nParInference));
    q0    = np.zeros((sys.nQInference,sys.T));
    q1    = np.zeros((classSMC.nPart,sys.nQInference));

    #=====================================================================
    # Main loop
    #=====================================================================
    for tt in range(0, sys.T-1):

        # Get the ancestor indicies
        at = classSMC.a[:,tt+1];
        at = at.astype(int);

        # Estimate state
        if ( classSMC.filterTypeInternal == "fullyadapted" ):
            xs[tt] = np.nansum( classSMC.p[:, tt] * classSMC.w[:, sys.T-2] );
        else:
            xs[tt] = np.nansum( classSMC.p[:, tt] * classSMC.w[:, sys.T-1] );

        #=================================================================
        # Estimate gradient
        #=================================================================
        if ( classSMC.calcGradientFlag ):

            # Estimate gradient and take the resampling into account
            g1 = g1[at,:] + sys.Dparm( classSMC.p[:,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

            # Calculate the gradient for each time step
            for nn in range(0,sys.nParInference):
                if ( ( classSMC.filterTypeInternal == "fullyadapted" ) & ( tt != sys.T-1 ) ):
                    g0[nn,tt]    = np.nansum( g1[:,nn] * classSMC.w[:,sys.T-1] );
                elif ( classSMC.filterTypeInternal != "fullyadapted" ):
                    g0[nn,tt]    = np.nansum( g1[:,nn] * classSMC.w[:,sys.T-1] );


        #=================================================================
        # Estimate of Q-function for EM algorithm
        #=================================================================
        if ( classSMC.calcQFlag ):
            oq1 = q1;

            if ( ( tt == sys.T-1 ) & ( classSMC.filterTypeInternal == "fullyadapted" ) ):
                # Do nothing
                at = 0;
            else:
                q1 = oq1[at,:] + sys.Qfunc( classSMC.p[:,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g0[:,sys.T-2]);
    calcHessian(classSMC,sys,g0);
    calcQfs(classSMC,sys,q1);

    # Save the smoothed state estimate
    classSMC.xhats = xs;
    classSMC.qfunc0 = q1;

##########################################################################
# Particle smoothing: fixed-lag smoother
##########################################################################
def proto_flPS(classSMC,sys,fixParticles=False):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "fl"
    setSettings(classSMC,"flsmoother");

    # Run initial filter
    if ( fixParticles == False ):
        classSMC.filter(sys);

    # Initalise variables
    xs    = np.zeros((sys.T,1));
    g1    = np.zeros((sys.nParInference,sys.T));
    h1    = np.zeros((sys.nParInference,sys.nParInference));
    h2    = np.zeros((sys.nParInference,sys.nParInference));
    sal   = np.zeros((classSMC.nPart,sys.nParInference,sys.T))
    q2    = np.zeros((sys.nQInference,sys.T));

    #=====================================================================
    # Main loop
    #=====================================================================

    # Construct the sa from the filter smoother to compute the cross-term in Louis identity
    if ( classSMC.calcHessianFlag == "louis" ):
        for tt in range(0, sys.T-1):

            # Get the ancestor indicies for the filter smoother
            bt = classSMC.a[:,tt+1]; bt = bt.astype(int);

            # Estimate the alpha quantity
            # Discussed in Dahlin, Lindsten, Schon (2014): Particle Metropolis-Hastings using gradient and Hessian information

            if ( classSMC.calcGradientFlag ):
                if ( classSMC.filterType == "abcPF" ):
                    sys.epsilon = classSMC.epsilon[tt];
                    sal[:,:,tt+1] = sal[bt,:,tt] + sys.Dparm( classSMC.p[:,tt+1], classSMC.p[bt,tt], classSMC.v1[bt,tt], classSMC.v2[bt,tt], tt );
                else:
                    sal[:,:,tt+1] = sal[bt,:,tt] + sys.Dparm( classSMC.p[:,tt+1], classSMC.p[bt,tt], np.zeros(classSMC.nPart), bt, tt );

    # Run the fixed-lag smoother for the rest
    for tt in range(0, sys.T-1):
        at  = np.arange(0,classSMC.nPart)
        kk  = np.min( (tt+classSMC.fixedLag, sys.T-1) )

        # Reconstruct particle trajectory
        for ii in range(kk,tt,-1):
            att = at.astype(int);
            at  = at.astype(int);
            at  = classSMC.a[at,ii];
            at  = at.astype(int);

        # Estimate state
        xs[tt] = np.nansum( classSMC.p[at,tt] * classSMC.w[:, kk] );

        #=================================================================
        # Estimate score and terms for the Louis identity
        #=================================================================

        if ( classSMC.calcGradientFlag ):
            if ( classSMC.filterType == "abcPF" ):
                sys.epsilon = classSMC.epsilon[tt];

                sa = sys.Dparm  ( classSMC.p[att,tt+1], classSMC.p[at,tt], classSMC.v1[at,tt], classSMC.v2[at,tt], tt);

                if ( classSMC.calcHessianFlag != False ):
                    sb = sys.DDparm ( classSMC.p[att,tt+1], classSMC.p[at,tt], classSMC.v1[at,tt], classSMC.v2[at,tt], tt);
            else:
                sa = sys.Dparm  ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

                if ( classSMC.calcHessianFlag != False ):
                    sb = sys.DDparm ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

            for nn in range(0,sys.nParInference):
                g1[nn,tt]       = np.nansum( sa[:,nn] * classSMC.w[:,kk] );

                if ( classSMC.calcHessianFlag == "louis" ):
                    for mm in range(0,(nn+1)):
                        h1[nn,mm] += np.nansum( classSMC.w[:,kk] * sb[:,nn,mm] );
                        h2[nn,mm] += np.nansum( classSMC.w[:,kk] * ( sa[:,nn]*sa[:,mm] + sa[:,nn]*sal[at,mm,tt] + sal[at,nn,tt]*sa[:,mm] ) );

                        h1[mm,nn]  = h1[nn,mm];
                        h2[mm,nn]  = h2[nn,mm];

        #=================================================================
        # Estimate of Q-function for EM algorithm
        #=================================================================
        if ( classSMC.calcQFlag  ):
            q1 = sys.Qfunc  ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);
            for nn in range(0,sys.nQInference):
                q2[nn,tt]       += np.nansum( q1[:,nn] * classSMC.w[:,kk] );

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1,h1,h2);
    calcQ(classSMC,sys,q2);
    classSMC.g1 = g1;

    # Save the smoothed state estimate
    classSMC.xhats = xs;
    classSMC.q2 = q2;

##########################################################################
# Particle smoother: forward-filter backward-smoother (FFBSm)
##########################################################################
def proto_ffbsmPS(classSMC,sys,fixParticles=False):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "ffbsm"
    setSettings(classSMC,"smoother");

    # Run initial filter
    if ( fixParticles == False ):
        classSMC.filter(sys);

    # Intialise variables
    v   = np.zeros((classSMC.nPart,sys.T));
    ws1 = np.zeros((classSMC.nPart,sys.T));
    ws2 = np.zeros((classSMC.nPart,classSMC.nPart,sys.T));
    xs  = np.zeros((sys.T,1));
    sa  = np.zeros((classSMC.nPart,sys.nParInference));
    sb  = np.zeros((classSMC.nPart,sys.nParInference,sys.nParInference));
    g1  = np.zeros((sys.nParInference,sys.T));
    q1  = np.zeros((sys.nQInference,sys.T));
    h1  = np.zeros((sys.nParInference,sys.nParInference));
    h2  = np.zeros((sys.nParInference,sys.nParInference));

    # Set initial weights and state
    ws1[:,sys.T-1] = classSMC.w[:,sys.T-1];
    xs[sys.T-1]    = classSMC.xhatf[sys.T-1];

    #=====================================================================
    # Main loop
    #=====================================================================
    for tt in range(sys.T-2,0,-1):
        if ( fixParticles == False ):
            # Compute the normalisation term
            for jj in range(0,classSMC.nPart):
                v[jj,tt] = np.nansum( classSMC.w[:,tt] * sys.evaluateState( classSMC.p[jj,tt+1], classSMC.p[:,tt], tt) );

        # Compute the smoothing weights
        for ii in range(0,classSMC.nPart):

            if ( fixParticles == False ):
                # Compute the 1-step smoothing weights
                ws1[ii,tt] = classSMC.w[ii,tt] * np.nansum( ws1[:,tt+1] * sys.evaluateState( classSMC.p[:,tt+1], classSMC.p[ii,tt], tt) / v[:,tt] )

            # Gradient and Hessian of the complete log-likelihood
            if ( classSMC.calcGradientFlag ):
                sa  = sys.Dparm(  classSMC.p[:,tt+1], classSMC.p[ii,tt], np.zeros(classSMC.nPart), range(0,classSMC.nPart), tt);

                if ( classSMC.calcHessianFlag != False ):
                    sb  = sys.DDparm( classSMC.p[:,tt+1], classSMC.p[ii,tt], np.zeros(classSMC.nPart), range(0,classSMC.nPart), tt);

                # Compute the 2-step smoothing weights
                if ( fixParticles != True ):
                    ws2[ii,:,tt] = classSMC.w[ii,tt] * ws1[:,tt+1] * sys.evaluateState( classSMC.p[:,tt+1], classSMC.p[ii,tt], tt) / v[:,tt]

                g1[:,tt] += np.dot( ws2[ii, :, tt], sa );

                if ( classSMC.calcHessianFlag == "louis" ):
                    h2       += np.tensordot(ws2[ii, :, tt] , sb, axes=([0],[0]));

                    for nn in range(0,sys.nParInference):
                        for mm in range(0,(nn+1)):
                            h1[nn,mm] += np.dot( ws2[ii, :, tt], sa[:,nn] * sa[:,mm] )
                            h1[mm,nn] = h1[nn,mm];

            # Q-function for EM
            if ( classSMC.calcQFlag  ):
                sq = sys.Qfunc  ( classSMC.p[:,tt+1], classSMC.p[ii,tt], np.zeros(classSMC.nPart), range(0,classSMC.nPart), tt);

                # Compute the 2-step smoothing weights
                if ( classSMC.calcGradientFlag != False ):
                    ws2[ii,:,tt] = classSMC.w[ii,tt] * ws1[:,tt+1] * sys.evaluateState( classSMC.p[:,tt+1], classSMC.p[ii,tt], tt) / v[:,tt]

                q1[:,tt] += np.dot( ws2[ii, :, tt], sq );

        # Estimate state
        xs[tt] = np.nansum( classSMC.p[:, tt] * ws1[:, tt] );

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1,h1,h2);
    calcQ(classSMC,sys,q1);

    # Save the smoothed state estimate
    classSMC.xhats = xs;

##########################################################################
# Particle smoother: forward-filter backward-simulator (FFBSi)
##########################################################################
def proto_ffbsiPS(classSMC,sys,fixParticles=False,rejectionSampling=True,earlyStopping=True):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "ffbsi"
    setSettings(classSMC,"ffbsismoother");

    # Run initial filter
    if ( fixParticles == False ):
        classSMC.filter(sys);

    # Intialise variables
    v  = np.zeros((classSMC.nPart,sys.T));
    ws = np.zeros((classSMC.nPart,1));
    xs = np.zeros((sys.T,1));
    sa = np.zeros((classSMC.nPart,sys.nParInference));
    sb = np.zeros((classSMC.nPart,sys.nParInference,sys.nParInference));
    g1 = np.zeros((sys.nParInference,sys.T));
    q1  = np.zeros((sys.nQInference,sys.T));
    h1 = np.zeros((sys.nParInference,sys.nParInference));
    h2 = np.zeros((sys.nParInference,sys.nParInference));

    if ( fixParticles == True ):
        ps = classSMC.ps;
    else:
        # Initialise the particle paths and weights
        ps            = np.zeros((classSMC.nPaths,sys.T));
        nIdx          = resampleMultinomial(classSMC.w[:,sys.T-1])[0:classSMC.nPaths];
        ps[:,sys.T-1] = classSMC.p[nIdx,sys.T-1];

        xs[sys.T-1]   = np.mean( ps[:,sys.T-1] );

    #=====================================================================
    # Main loop
    #=====================================================================
    for tt in range(sys.T-2,0,-1):

        if ( fixParticles == False ):
            if ( rejectionSampling == False ):
                # Use standard formulation of FFBSi

                for jj in range(0,classSMC.nPaths):
                    # Compute the normalisation term
                    v[jj,tt] = np.nansum( classSMC.w[:,tt] * sys.evaluateState( classSMC.p[jj,tt+1], classSMC.p[:,tt], tt) );

                    # Compute the 1-step smoothing weights
                    ws = classSMC.w[:,tt] * sys.evaluateState( ps[jj,tt+1], classSMC.p[:,tt], tt) / v[jj,tt];

                    # Sample from the backward kernel
                    pIdx = resampleMultinomial(ws)[0];

                    # Append the new particle
                    ps[jj,tt]  = classSMC.p[pIdx,tt];

            else:
                # Use rejection sampling
                L           = np.arange( classSMC.nPaths ).astype(int);

                if ( earlyStopping ):
                    counterLimit = classSMC.nPathsLimit;
                else:
                    counterLimit = classSMC.nPaths * 100;

                counterIter = 0;

                # As long as we have trajectories left to sample and have not reach the early stopping
                while ( ( len(L) > 0 ) & ( counterIter < counterLimit ) ):

                    # Compute the length of L
                    n = len(L);

                    # Sample n weights and uniforms
                    I = np.random.choice( classSMC.nPart, p=classSMC.w[:,tt], size=n  )
                    U = np.random.uniform( size=n  )

                    # Compute the acceptance probability and the decision
                    prob   = sys.evaluateState( ps[L,tt+1], classSMC.p[I,tt], tt);
                    accept = (U <= (prob / classSMC.rho) );

                    # Append the new particle to the trajectory
                    ps[ L[accept], tt ]  = classSMC.p[ I[accept], tt ];

                    # Remove the accepted elements from the list
                    L = np.delete( L, np.where( accept == True ) );

                    counterIter += 1;

                # Print error message if we have reached the limit and do not have early stopping
                if ( ( earlyStopping == False) & ( counterIter == counterLimit ) ):
                    raise NameError("To many iterations, aborting");


                # Use standard FFBSi for the remaing trajectories if we have early stopping
                if ( ( earlyStopping == True) & ( counterIter == counterLimit ) ):

                    for jj in L:
                        # Compute the normalisation term
                        v[jj,tt] = np.nansum( classSMC.w[:,tt] * sys.evaluateState( classSMC.p[jj,tt+1], classSMC.p[:,tt], tt) );

                        # Compute the 1-step smoothing weights
                        ws = classSMC.w[:,tt] * sys.evaluateState( ps[jj,tt+1], classSMC.p[:,tt], tt) / v[jj,tt];

                        # Sample from the backward kernel
                        pIdx = resampleMultinomial(ws)[0];

                        # Append the new particle
                        ps[jj,tt]  = classSMC.p[pIdx,tt];

            # Estimate state
            xs[tt] = np.mean( ps[:, tt] );

        if ( classSMC.calcGradientFlag ):
            # Gradient and Hessian of the complete log-likelihood
            sa  = sys.Dparm(  ps[:,tt+1], ps[:,tt], np.zeros(classSMC.nPaths), range(0,classSMC.nPaths), tt);
            sb  = sys.DDparm( ps[:,tt+1], ps[:,tt], np.zeros(classSMC.nPaths), range(0,classSMC.nPaths), tt);

            g1[:,tt] = np.mean( sa, axis=0);

            if ( classSMC.calcHessianFlag == "louis" ):
                h2       += np.mean( sb, axis=0);

                for nn in range(0,sys.nParInference):
                    for mm in range(0,(nn+1)):
                        h1[nn,mm] += np.mean( sa[:,nn] * sa[:,mm], axis=0 )
                        h1[mm,nn] = h1[nn,mm];

        # Q-function for EM
        if ( classSMC.calcQFlag  ):
            sq = sys.Qfunc  ( ps[:,tt+1], ps[:,tt], np.zeros(classSMC.nPart), range(0,classSMC.nPaths), tt);
            q1[:,tt] = np.mean( sq, axis=0 );

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1,h1,h2);
    calcQ(classSMC,sys,q1);

    # Save the smoothed state estimate
    classSMC.xhats = xs;
    if ( fixParticles == False ):
        classSMC.ps    = ps;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################