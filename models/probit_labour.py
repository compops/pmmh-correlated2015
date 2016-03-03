##############################################################################
##############################################################################
# Model specification
# Generalised Multinomial Logit model
# Version 2015-10-30
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy          as     np
import weave

from  models_helpers  import *
from  models_dists    import *

class ssm(object):
    #=========================================================================
    # Define model settings
    #=========================================================================
    modelName    = "Probit model"
    filePrefix   = "probit-labour";
    typeSampler  = "exact";

    nRegressors   = 7;
    T             = 753;
    nPar          = 7 + 1 + 1;

    par           = np.zeros( ( nPar, 1 ), dtype='double' )
    par[0]        =  0.5855; # beta0
    par[1]        = -0.0034; # beta1
    par[2]        =  0.0380; # beta2
    par[3]        =  0.0395; # beta3
    par[4]        = -0.0006; # beta4
    par[5]        = -0.0161; # beta5
    par[6]        = -0.2618; # beta6
    par[7]        =  0.0130; # beta7
    par[8]        =  1.0000; # sigma
    xo            =  0.0;

    #=============================================================================
    # Store the parameters into the struct
    #=============================================================================
    def storeParameters( self, newParm, sys ):
        self.par = np.zeros( sys.nPar, dtype='double' );

        for kk in range(0,self.nParInference):
            self.par[kk] = np.array( newParm[kk], copy=True, dtype='double' )

        for kk in range(self.nParInference,sys.nPar):
            self.par[kk] = sys.par[ kk ];

    #=============================================================================
    # Returns the current parameters stored in this struct
    #=============================================================================
    def returnParameters(self):
        out = np.zeros( self.nParInference, dtype='double');

        for kk in range(0,self.nParInference):
            out[kk]  = self.par[ kk ];

        return(out);

    #=============================================================================
    # Load data
    #=============================================================================
    def loadData( self, fileName ):
        data     = np.loadtxt( fileName );

        self.y   = data[:,1].astype(np.double);
        xin      = data[:,(19,5,18,18,4,2,3)];

        self.x       = np.ones( ( self.T,self.nRegressors+1 ), dtype='double' )
        self.u       = 0.0;
        self.x[:,1:(self.nRegressors+1)] = xin;
        self.x[:,4]  = self.x[:,4]**2;

    #=========================================================================
    # Estimate the log-likelihood
    #=========================================================================
    def estimateLogLikelihood( self, model ):

        ###############################################################################
        ## Estimate log-likelihood using importance sampling by
        ## marginalise over the latent variables simulated from the prior/model
        ###############################################################################

        importanceSampler = \
        """
        double foo = 0.0;

        //================================================================
        // Compute the latents for each individual
        //================================================================
        for ( int tt = 0; tt < cT; ++tt )
        {
            for ( int kk = 0; kk < ( cnRegressors + 1); ++kk )
            {
                cutility( tt ) += cpar( kk ) * cx( tt, kk );
            }
        }

        //====================================================================
        // Importance sampling with nPart samples
        //====================================================================
        for ( int nn = 0; nn < cnPart; ++nn )
        {
            for ( int tt = 0; tt < cT; ++tt )
            {
                foo = cutility( tt ) + cepsilon( nn, tt );

                if ( foo >= 0.0 ) {
                    cps( tt ) += 1.0 / (double) cnPart;
                }
            }
        }

        //====================================================================
        // Compute log-likelihood estimate
        //====================================================================

        for ( int tt = 0; tt < cT; ++tt )
        {
            if ( cy( tt ) > 0.5 ) {

                // y_t=1
                cllt(tt) = log( cps( tt ) );

            } else {

                // y_t=0
                cllt(tt) = log( 1.0 - cps( tt ) );
            }
        }
        """

        # Random variables
        self.epsilon = self.rv; # nPart * T;
        self.nPart   = ( self.rv ).shape[0];

        # Pre-allocate variables
        self.utility = np.zeros( self.T, dtype='double' )
        self.ps      = np.zeros( self.T, dtype='double' )
        self.llt     = np.zeros( self.T, dtype='double' )

        my_dict = {
            'cutility':self.utility,
            'cnRegressors':self.nRegressors,
            'cT':self.T,
            'cpar':self.par,
            'cx':self.x,
            'cy':self.y,
            'cepsilon':self.epsilon,
            'cps':self.ps,
            'cnPart':self.nPart,
            'cllt':self.llt
        }

        weave.inline(importanceSampler,['cutility','cnRegressors','cT','cpar','cx','cy','cepsilon','cps','cnPart','cllt'],  local_dict=my_dict, type_converters=weave.converters.blitz, headers = ['<math.h>','<stdio.h>'], compiler = 'gcc' )
        self.ll     = np.nansum( self.llt );
        self.xtraj  = np.zeros( self.T );

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        return( 1.0 );

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================
    def prior(self):
        out = 0.0;

        mu = np.array( ( 0.5855, -0.0034, 0.0380, 0.0395, -0.0006, -0.0161, -0.2618, 0.0130 ) )[ range(self.nParInference) ]

        # Normal prior for beta
        out += MultivariateNormalLogPDF( self.par[ range(self.nParInference) ], mu, np.diag( np.ones( self.nParInference ) ) )

        # Uniform for sigmae
        out += 0.0;

        return out;

    #=============================================================================
    # Copy data from an instance of this struct to another
    #=============================================================================
    def copyData( self, sys ):

        self.nRegressors  = sys.nRegressors
        self.T   = sys.T
        self.nPar         = sys.nPar
        self.T            = sys.T;

        self.y             = np.copy( sys.y )
        self.u             = np.copy( sys.u )
        self.x             = np.copy( sys.x )

        # Copy parameters
        self.par = np.zeros( sys.nPar );

        for kk in range( sys.nPar ):
            self.par[kk] = np.array( sys.par[kk], copy=True )

    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

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