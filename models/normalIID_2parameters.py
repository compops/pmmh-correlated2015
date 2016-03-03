##############################################################################
##############################################################################
# Model specification
# Normal IID model
# Version 2015-05-08
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy          as     np
from   scipy.stats    import norm
from   models_helpers import *
from   models_dists   import *


class ssm(object):
    #=========================================================================
    # Define model settings
    #=========================================================================
    nPar         = 3;
    par          = np.zeros(nPar);
    modelName    = "Normal IID model with two parameters"
    filePrefix   = "normalIID";
    supportsFA   = False;
    nParInfernce = None;
    nQInference  = None;

    #=========================================================================
    # Define the model
    #=========================================================================
    def generateInitialState( self, nPart ):
        return self.par[0] + self.par[1] * np.random.normal(size=(1,nPart));

    def generateState(self, xt, tt):
        return self.par[0] + self.par[1] * np.random.randn(1,len(xt));

    def generateInitialStateRV( self, nPart, u ):
        return self.par[0] + self.par[1] * u[ range(1,nPart+1) ];

    def generateStateRV(self, xt, tt, u):
        return self.par[0] + self.par[1] * u[ range(1,len(xt)+1) ];

    def evaluateState(self, xtt, xt, tt):
        return norm.pdf( xtt, self.par[0], self.par[1]  );

    def generateObservation(self, xt, tt):
        return xt + self.par[2] * np.random.randn(1,len(xt));

    def evaluateObservation(self, xt, tt):
        return norm.logpdf(self.y[tt], xt, self.par[2] );

    #=========================================================================
    # Define gradients of logarithm of complete data-likelihood
    #=========================================================================
    def Dparm(self, xtt, xt, pu, pw, tt):
        nOut = len(xtt);
        gradient = np.zeros(( nOut, self.nParInference ));
        return(gradient);

    #=========================================================================
    # Define Hessians of logarithm of complete data-likelihood
    #=========================================================================
    def DDparm(self, xtt, xt, st, at, tt):

        nOut = len(xtt);
        hessian = np.zeros( (nOut, self.nParInference,self.nParInference) );
        return(hessian);

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0;

        if( self.par[1] < 0.0 ):
            out = 0.0;

        return( out );

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================
    def prior(self):
        out = 0.0;

        # Truncated normal prior for mu
        if ( self.nParInference >= 1 ):
            out += normalLogPDF( self.par[0], 0.0, 1.0 );

        # Gamma prior for sigma
        if ( self.nParInference >= 2 ):
            out += gammaLogPDF( self.par[1], a=2.0, b=4.0 );

        return out;


    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Standard operations on struct
    copyData                = template_copyData;
    storeParameters         = template_storeParameters;
    returnParameters        = template_returnParameters

    # No tranformations available
    transform               = empty_transform;
    invTransform            = empty_invTransform;
    Jacobian                = empty_Jacobian;

    # Standard data generation for this model
    generateData            = template_generateData;

    # No faPF available for this model
    generateStateFA         = empty_generateStateFA;
    evaluateObservationFA   = empty_evaluateObservationFA;
    generateObservationFA   = empty_generateObservationFA;

    # No EM algorithm available for this model
    Qfunc                   = empty_Qfunc;
    Mstep                   = empty_Mstep;