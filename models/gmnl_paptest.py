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
    modelName    = "Generalised Multinomial Logit model"
    filePrefix   = "gmnl-paptest";
    typeSampler  = "exact";
    version      = "transformed"

    nIndividuals  = 79;
    nAttributes   = 5;
    nOccasions    = 32;
    nPar          = 4 + 2 * 5 -1;
    T             = 1;

    par           = np.zeros( ( nPar, 1 ), dtype='double' )
    par[0]        = -1.183; # beta01
    par[1]        =  1.092; # beta1
    par[2]        = -1.763; # beta2
    par[3]        =  4.091; # beta3
    par[4]        =  1.658; # beta4
    par[5]        = -0.271; # beta5
    par[6]        =  3.121; # sigma01
    par[7]        =  1.693; # sigma1
    par[8]        =  2.856; # sigma2
    par[9]        =  3.005; # sigma3
    par[10]       =  1.247; # sigma4
    par[11]       =  0.621; # delta
    par[12]       =  0.170; # gamma
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

        self.parBeta0            = float( self.par[0] )
        self.parSigma0           = float( self.par[6] )
        self.parDelta            = float( self.par[11] )
        self.parGamma            = float( self.par[12] )

        self.parSigma            = np.zeros( 5, dtype='double' )
        self.parSigma[ 0:4 ]     = self.par[ range( 7, 11 ) ];
        self.parBeta             = np.zeros( 5, dtype='double' )
        self.parBeta             = self.par[ range( 1, 6 ) ];

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
        n        = data.shape[0];
        oddindex = ( np.arange(0,n,2) ).astype(int)

        yin      = data[oddindex,3]
        xin      = data[oddindex,5:]

        self.x    = np.zeros( ( self.nIndividuals, self.nAttributes, self.nOccasions  ), dtype='double' )
        self.y    = np.zeros( ( self.nIndividuals, self.nOccasions                    ), dtype='double' )
        self.u    = np.zeros( ( self.nIndividuals, self.nOccasions                    ), dtype='double' )

        for ii in range( self.nIndividuals ):
            idx                 = range(ii*self.nOccasions,(ii+1)*self.nOccasions0,);
            self.x[ ii, : , : ] = xin[ idx, :  ].transpose()
            self.y[ ii, :    ]  = yin[ idx ]

    #=========================================================================
    # Define Jacobian and parameter transforms
    #=========================================================================
    def Jacobian( self ):

        JacobOut = 0.0;

        if (self.version == "transformed"):

            # Sigma01
            if ( self.nParInference > 6 ):
                JacobOut += np.log( self.parSigma0 );

            # Sigma1
            if ( self.nParInference > 7 ):
                JacobOut += np.log( self.parSigma[0] );

            # Sigma2
            if ( self.nParInference > 8 ):
                JacobOut += np.log( self.parSigma[1] );

            # Sigma 3
            if ( self.nParInference > 9 ):
                JacobOut += np.log( self.parSigma[2] );

            # Sigma 4
            if ( self.nParInference > 10 ):
                JacobOut += np.log( self.parSigma[3] );

            # Delta
            if ( self.nParInference > 11 ):
                JacobOut += np.log( self.parDelta );

            # Gamma
            if ( self.nParInference > 12 ):
                JacobOut += np.log( self.parGamma - self.parGamma**2 );

        return JacobOut;

    def transform(self):
        if (self.version == "transformed"):

            # Sigma01 ( positive )
            if ( self.nParInference > 6 ):
                self.par[6]     = np.exp ( self.par[6] );
                self.parSigma0  = self.par[6];

            # Sigma1 ( positive )
            if ( self.nParInference > 7 ):
                self.par[7]      = np.exp ( self.par[7] );
                self.parSigma[0] = self.par[7];

            # Sigma2 ( positive )
            if ( self.nParInference > 8 ):
                self.par[8]      = np.exp ( self.par[8] );
                self.parSigma[1] = self.par[8];

            # Sigma3 ( positive )
            if ( self.nParInference > 9 ):
                self.par[9]      = np.exp ( self.par[9] );
                self.parSigma[2] = self.par[9];

            # Sigma4 ( positive )
            if ( self.nParInference > 10 ):
                self.par[10]     = np.exp ( self.par[10] );
                self.parSigma[3] = self.par[10];

            # Delta ( positive )
            if ( self.nParInference > 11 ):
                self.par[11]  = np.exp ( self.par[11] );
                self.parDelta = self.par[11];

            # Gamma ( Within the unit interval )
            if ( self.nParInference > 12 ):
                self.par[12]  = invlogit( self.par[12] );
                self.parGamma = self.par[12];
        return None;

    def invTransform(self):
        if (self.version == "transformed"):

            # Sigma01
            if ( self.nParInference > 6 ):
                self.par[6]     = np.log ( self.par[6] );
                self.parSigma0  = self.par[6];

            # Sigma1
            if ( self.nParInference > 7 ):
                self.par[7]      = np.log ( self.par[7] );
                self.parSigma[0] = self.par[7];

            # Sigma2
            if ( self.nParInference > 8 ):
                self.par[8]      = np.log ( self.par[8] );
                self.parSigma[1] = self.par[8];

            # Sigma3
            if ( self.nParInference > 9 ):
                self.par[9]      = np.log ( self.par[9] );
                self.parSigma[2] = self.par[9];

            # Sigma4
            if ( self.nParInference > 10 ):
                self.par[10]     = np.log ( self.par[10] );
                self.parSigma[3] = self.par[10];

            # Delta
            if ( self.nParInference > 11 ):
                self.par[11]  = np.log ( self.par[11] );
                self.parDelta = self.par[11];

            # Gamma
            if ( self.nParInference > 12 ):
                self.par[12]  = logit( self.par[12] );
                self.parGamma = self.par[12];
        return None;

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
        double logWeight        = 0.0;
        double logLikelihood    = 0.0;
        double logLikelihoodInv = 0.0;
        double logProb          = 0.0;

        double foo1             = 0.0;
        double foo2             = 0.0;
        double foo3             = 0.0;

        double lam              = 0.0;
        double beta0            = 0.0;
        double beta [ 5 ];

        int nOccasions          = 32;
        int nIndividuals        = 79;
        int nAttributes         = 5;

        //====================================================================
        // Importance sampling with nPart samples over each individual
        //====================================================================
        for ( int ii = 0; ii < nIndividuals; ++ii )
        {

            logLikelihoodInv = 0.0;

            //================================================================
            // Compute the latents for each sample
            // Compute the resulting weight
            //================================================================
            for ( int nn = 0; nn < cnPart; ++nn )
            {
                // Scaling coefficient \lambda_{ii}
                lam = exp( -0.5 * cparDelta * cparDelta + cparDelta  * cxi( ii, nn ) );

                // Intercept \beta_{0i}
                beta0 = cparBeta0  + cparSigma0 * ceta0( ii, nn );

                // Regressor coefficients \beta_{ki}
                for ( int kk = 0; kk < nAttributes; ++kk )
                {
                    foo1    = lam * cparBeta( kk );
                    foo2    = cparGamma * cparSigma( kk ) * ceta( ii, kk, nn );
                    foo3    = ( 1.0 - cparGamma ) * lam * cparSigma( kk ) * ceta( ii, kk, nn );

                    beta[ kk ] = foo1 + foo2 + foo3;
                }

                // For each occasion
                for ( int tt = 0; tt < nOccasions; ++tt )
                {
                    logProb = 0.0;

                    // Compute the log-probability of individual ii to take the test at time tt
                    logProb = beta0;

                    for ( int kk = 0; kk < nAttributes; ++kk )
                    {
                        logProb += beta[ kk ] * cx( ii, kk, tt );
                    }

                    // Compute the unnormalised weight
                    logLikelihoodInv += exp( cy( ii, tt ) * logProb - log( 1.0 + exp( logProb ) ) );
                }
            }

            logLikelihood += log( logLikelihoodInv ) - nOccasions * log( cnPart );
        }

        return_val = logLikelihood;
        """

        # Random variables
        self.xi     = self.rv[ range( self.nIndividuals*self.nPart ), 0 ];
        self.eta0   = self.rv[ range( self.nIndividuals*self.nPart, 2*self.nIndividuals*self.nPart ), 0 ];
        self.eta    = self.rv[ range( 2*self.nIndividuals*self.nPart, ( self.rv[ :, 0 ] ).shape[0] ), 0 ];

        self.xi     = (self.xi).reshape(   ( self.nIndividuals, self.nPart   ) );
        self.eta0   = (self.eta0).reshape( ( self.nIndividuals, self.nPart   ) );
        self.eta    = (self.eta).reshape(  ( self.nIndividuals, self.nAttributes,  self.nPart   ) );

        # Pre-allocate variables
        self.beta0  = np.zeros( ( self.nPart                                 ), dtype='double' )
        self.beta   = np.zeros( ( self.nPart, self.nAttributes               ), dtype='double' )
        self.lam    = np.zeros( ( self.nPart                                 ), dtype='double' )
        self.ps     = np.zeros( ( self.nIndividuals, self.nOccasions,  self.nPart   ), dtype='double' )

        my_dict = {
            'cparBeta0':self.parBeta0,
            'cparSigma0':self.parSigma0,
            'cparBeta':self.parBeta,
            'cparSigma':self.parSigma,
            'cparDelta':self.parDelta,
            'cparGamma':self.parGamma,
            'cx':self.x,
            'cy':self.y,
            'ceta0':self.eta0,
            'ceta':self.eta,
            'cxi':self.xi,
            'cnPart':self.nPart,
        }

        self.ll = weave.inline(importanceSampler,['cparBeta0','cparSigma0','cparBeta','cparSigma','cparDelta','cparGamma','cx','cy','ceta0','ceta','cxi','cnPart'],  local_dict=my_dict, type_converters=weave.converters.blitz, headers = ['<math.h>','<stdio.h>'], compiler = 'gcc' )

        self.xtraj  = np.zeros( self.T );

    #=========================================================================
    # Estimate the log-likelihood
    #=========================================================================
    def estimateLogLikelihood_robust( self, model ):

        ###############################################################################
        ## Estimate log-likelihood using importance sampling by
        ## marginalise over the latent variables simulated from the prior/model
        ###############################################################################

        importanceSampler = \
        """
        double logProb = 0.0;
        double foo1 = 0.0;
        double foo2 = 0.0;
        double foo3 = 0.0;
        double foo4 = 0.0;

        //====================================================================
        // Importance sampling with nPart samples
        //====================================================================
        for ( int nn = 0; nn < cnPart; ++nn )
        {
            //================================================================
            // Compute the latents for each individual
            //================================================================
            for ( int ii = 0; ii < cnIndividuals; ++ii )
            {
                // Scaling coefficient \lambda_{ii}
                foo1       = -0.5 * ( (double) cparDelta ) * ( (double) cparDelta );
                foo2       = ( (double) cparDelta ) * ( (double) cxi( ii, nn ) );
                clam( ii ) =  exp( foo1 + foo2 );

                // Intercept \beta_{0i}
                cbeta0( ii ) = ( (double) cparBeta0  ) + ( (double) cparSigma0 ) * ( (double) ceta0( ii, nn ) );

                // Regressor coefficients \beta_{ki}
                for ( int kk = 0; kk < cnAttributes; ++kk )
                {
                    foo1    = ( (double) clam( ii ) )  * ( (double) cparBeta( kk ) );
                    foo2    = ( (double) cparGamma  )  * ( (double) cparSigma( kk ) ) * ( (double) ceta( ii, kk, nn ) );
                    foo3    = ( 1.0 - cparGamma )      * ( (double) clam( ii ) );
                    foo4    = (double) cparSigma( kk ) * ( (double) ceta( ii, kk, nn ) );
                    cbeta( ii, kk ) = foo1 + foo2 + foo3 * foo4;
                }
            }

            //================================================================
            // Compute weights
            //================================================================

            // For each occasion
            for ( int tt = 0; tt < cnOccasions; ++tt )
            {
                logProb = 0.0;

                // For each individual
                for( int ii = 0; ii < cnIndividuals; ++ii )
                {
                    // Compute the log-probability of individual ii to take the test at time tt
                    logProb = (double) cbeta0( ii );

                    for ( int kk = 0; kk < cnAttributes; ++kk )
                    {
                        logProb += (double) cbeta( ii, kk ) * (double) cx( ii, kk, tt );
                    }

                    // Compute the unnormalised weight
                    if ( logProb > 0.0 ) {

                        // Positive utillity

                        if ( cy( ii, tt) > 0.5 ) {
                            // y_{it} = 1
                            cps( ii, tt, nn ) = - log( 1.0 + exp( -logProb ) );

                        } else {
                            // y_{it} = 0
                            cps( ii, tt, nn ) = - logProb - log( 1.0 + exp( -logProb ) );
                        }
                    } else {

                        // Negative utillity

                        if ( cy( ii, tt) > 0.5 ) {
                            // y_{it} = 1
                            cps( ii, tt, nn ) = logProb - log( 1.0 + exp( logProb ) );

                        } else {
                            // y_{it} = 0
                            cps( ii, tt, nn ) = -log( 1.0 + exp( logProb ) );
                        }
                    }

                    // Find maximum element
                    if ( cps( ii, tt, nn ) > cwmax( ii, tt ) )
                    {
                        cwmax( ii, tt ) = cps( ii, tt, nn );
                    }
                }
            }
        }

        //====================================================================
        // Compute log-likelihood estimate
        //====================================================================

        double tmpSum = 0.0;
        double ll     = 0.0;
        double foo    = 0.0;

        for( int ii = 0; ii < cnIndividuals; ++ii )
        {
            for ( int tt = 0; tt < cnOccasions; ++tt )
            {
                tmpSum = 0.0;

                for ( int nn = 0; nn < cnPart; ++nn )
                {
                    tmpSum += exp( cps( ii, tt, nn ) - cwmax( ii, tt ) );
                }

                ll += cwmax( ii, tt ) + log( tmpSum ) - log( cnPart );
            }
        }

        return_val = ll;
        """

        # Random variables
        self.xi     = self.rv[ range( self.nIndividuals*self.nPart ), 0 ];
        self.eta0   = self.rv[ range( self.nIndividuals*self.nPart, 2*self.nIndividuals*self.nPart ), 0 ];
        self.eta    = self.rv[ range( 2*self.nIndividuals*self.nPart, ( self.rv[ :, 0 ] ).shape[0] ), 0 ];

        self.xi     = (self.xi).reshape(   ( self.nIndividuals, self.nPart   ) );
        self.eta0   = (self.eta0).reshape( ( self.nIndividuals, self.nPart   ) );
        self.eta    = (self.eta).reshape(  ( self.nIndividuals, self.nAttributes,  self.nPart   ) );

        # Pre-allocate variables
        self.beta0  = np.zeros( ( self.nIndividuals                                 ), dtype='double' )
        self.beta   = np.zeros( ( self.nIndividuals, self.nAttributes               ), dtype='double' )
        self.lam    = np.zeros( ( self.nIndividuals                                 ), dtype='double' )
        self.ps     = np.zeros( ( self.nIndividuals, self.nOccasions,  self.nPart   ), dtype='double' )
        self.wmax   = -100000000.0 * np.ones(  ( self.nIndividuals, self.nOccasions                ), dtype='double' )

        my_dict = {
            'cnIndividuals':self.nIndividuals,
            'cnAttributes':self.nAttributes,
            'cnOccasions':self.nOccasions,
            'cparBeta0':self.parBeta0,
            'cparSigma0':self.parSigma0,
            'cparBeta':self.parBeta,
            'cparSigma':self.parSigma,
            'cparDelta':self.parDelta,
            'cparGamma':self.parGamma,
            'cx':self.x,
            'cy':self.y,
            'ceta0':self.eta0,
            'ceta':self.eta,
            'cxi':self.xi,
            'cbeta0':self.beta0,
            'cbeta':self.beta,
            'clam':self.lam,
            'cps':self.ps,
            'cnPart':self.nPart,
            'cwmax':self.wmax
        }

        self.ll = weave.inline(importanceSampler,['cnIndividuals','cnAttributes','cnOccasions','cparBeta0','cparSigma0','cparBeta','cparSigma','cparDelta','cparGamma','cx','cy','ceta0','ceta','cxi','cbeta0','cbeta','clam','cps','cnPart','cwmax'],  local_dict=my_dict, type_converters=weave.converters.blitz, headers = ['<math.h>','<stdio.h>'], compiler = 'gcc' )

        self.xtraj  = np.zeros( self.T );

    #=========================================================================
    # Estimate the log-likelihood
    #=========================================================================
    def estimateLogLikelihood_python( self, model ):

        ###############################################################################
        ## Estimate log-likelihood using importance sampling by
        ## marginalise over the latent variables simulated from the prior/model
        ###############################################################################

        logLikelihood  = 0;

        # Random variables
        self.xi     = self.rv[ range( self.nIndividuals*self.nPart ), 0 ];
        self.eta0   = self.rv[ range( self.nIndividuals*self.nPart, 2*self.nIndividuals*self.nPart ), 0 ];
        self.eta    = self.rv[ range( 2*self.nIndividuals*self.nPart, ( self.rv[ :, 0 ] ).shape[0] ), 0 ];

        self.xi     = (self.xi).reshape(   ( self.nIndividuals, self.nPart   ) );
        self.eta0   = (self.eta0).reshape( ( self.nIndividuals, self.nPart   ) );
        self.eta    = (self.eta).reshape(  ( self.nIndividuals, self.nAttributes,  self.nPart   ) );

        # Pre-allocate variables
        self.beta0  = np.zeros( ( self.nIndividuals                                 ), dtype='double' )
        self.beta   = np.zeros( ( self.nIndividuals, self.nAttributes               ), dtype='double' )
        self.lam    = np.zeros( ( self.nIndividuals                                 ), dtype='double' )
        self.ps     = np.zeros( ( self.nIndividuals, self.nOccasions                ), dtype='double' )

        for nn in range( self.nPart ):

            #------------------------------------------------------------------------------
            # Sample latent variables from the prior
            #------------------------------------------------------------------------------

            for ii in range( self.nIndividuals ):

                # Sample the scaling
                self.lam[ ii ]   = np.exp( -0.5 * self.parDelta**2 + self.parDelta * self.xi[ ii, nn ] );

                # Sample individual intercept
                self.beta0[ ii ] = self.parBeta0 + self.parSigma0 * self.eta0[ ii, nn ];

                # Compute the individual scalings of the attributes
                for kk in range( self.nAttributes ):
                    foo1                = self.lam[ ii ] * self.parBeta[ kk ] + self.parGamma * self.parSigma[ kk ] * self.eta[ ii, kk, nn ];
                    foo2                = ( 1.0 - self.parGamma ) * self.lam[ ii ]            * self.parSigma[ kk ] * self.eta[ ii, kk, nn ];
                    self.beta[ ii, kk ] = foo1 + foo2;

            #------------------------------------------------------------------------------
            # Estimate the log-likelihood
            #------------------------------------------------------------------------------

            for ii in range( self.nIndividuals ):
                for tt in range( self.nOccasions ):

                    # Compute the probability of individual ii to select outcome jj at time tt
                    logProb  = self.beta0[ ii ] + np.sum( self.beta[ ii, : ] * self.x[ :, ii, tt ] );

                    # Normalise the probability over all outcomes
                    self.ps[ ii, tt ] = self.y[ ii, tt ] * logProb - np.log( 1.0 + np.exp( logProb ) );

            # Compute the contribution to the estimate of the log-likelihood
            wmax           = np.max( self.ps );
            weights        = np.exp( self.ps - wmax );
            logLikelihood += wmax * np.log( np.sum( weights ) ) - np.log ( self.nPart );

        self.ll     = logLikelihood;
        self.xtraj  = np.zeros( self.T );

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):

        out = 1.0;

        if (self.version != "transformed"):
            for ii in range( 6, 12 ):
                if ( self.par[ii] < 0.0 ):
                    out = 0.0;

            if ( ( self.par[12] < 0.0 ) | ( self.par[12] > 1.0 ) ):
                out = 0.0;

        return( out );

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================
    def prior(self):
        out = 0.0;

        # Normal prior for beta01
        out += normalLogPDF( self.parBeta0, 0.0, 1.0 );

        # Gamma prior for sigma01
        out += gammaLogPDF( self.parSigma0, a=2.0, b=2.0 )

        # Normal prior for betak
        out += np.sum( normalLogPDF( self.parBeta,  0.0, 3.0 ) );

        # Gamma prior prior for sigmak
        out += np.sum( gammaLogPDF( self.parSigma[0:4], a=2.0, b=1.0 ) );

        if ( self.nParInference > 11 ):
            # ?? for delta
            out += gammaLogPDF( self.parDelta, a=0.5, b=1.0 );

        if ( self.nParInference > 12 ):
            # Uniform for gamma
            out += 0.0;

        return out;

#        # Normal prior for beta01
#        out += normalLogPDF( self.parBeta0, 0.0, 10.0 );
#
#        # Half-Cauchy prior for sigma01
#        out += halfCauchyLogPDF( self.parSigma0, 0.0, 1.0)
#
#        # Normal prior for betak
#        out += np.sum( normalLogPDF    ( self.parBeta,  0.0, 10.0 ) );
#
#        # Half-Cauchy prior for sigmak
#        out += np.sum( halfCauchyLogPDF( self.parSigma[0:4], 0.0, 1.0 )  );
#
#        # ?? for delta
#        out += -np.log( 1 + self.parDelta / 0.2 );
#
#        # Uniform for gamma
#        out += 0.0;
#
#        return out;

    #=============================================================================
    # Copy data from an instance of this struct to another
    #=============================================================================
    def copyData( self, sys ):

        self.nIndividuals = sys.nIndividuals
        self.nAttributes  = sys.nAttributes
        self.nOccasions   = sys.nOccasions
        self.nPar         = sys.nPar

        self.y             = np.copy( sys.y )
        self.u             = np.copy( sys.u )
        self.x             = np.copy( sys.x )
        self.T             = sys.T;

        # Copy parameters
        self.par = np.zeros( sys.nPar );

        for kk in range( sys.nPar ):
            self.par[kk] = np.array( sys.par[kk], copy=True )

    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Standard data generation for this model
    generateData            = template_generateData;

    # No faPF available for this model
    generateStateFA         = empty_generateStateFA;
    evaluateObservationFA   = empty_evaluateObservationFA;
    generateObservationFA   = empty_generateObservationFA;

    # No EM algorithm available for this model
    Qfunc                   = empty_Qfunc;
    Mstep                   = empty_Mstep;