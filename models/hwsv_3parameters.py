##############################################################################
##############################################################################
# Model specification
# Simple HW model
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

#=============================================================================
# Model structure
#=============================================================================
#
# xtt = par[0] + par[1] * ( xt - par[0] ) + par[2] * vt,    
# yt  = exp( 0.5 * xt) * et,
# 
# ( v(t), e(t) ) ~ N(0,E), E = ( 1, par[3]; par[3], 1)

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
    modelName    = "Hull-White stochastic volatility with three parameters"
    filePrefix   = "hwsv";
    supportsFA   = False;
    nParInfernce = None;
    nQInference  = None;

    #=========================================================================
    # Define the model
    #=========================================================================
    def generateInitialState( self, nPart ):
        return np.random.normal(size=(1,nPart)) * self.par[1] / np.sqrt( 1 - self.par[0]**2 );

    def evaluateState(self, xtt, xt, tt):
        return norm.pdf( xtt, self.par[0] * xt, self.par[1] );       
    
    def generateState(self, xt, tt):
        return self.par[0] * xt + self.par[1] * np.random.randn(1,len(xt));
      
    def generateObservation(self, xt, tt):
        return np.random.randn(1,len(xt)) * self.par[2] * np.exp( 0.5 * xt ) ;
    
    def evaluateObservation(self, xt, tt):
        return norm.logpdf(self.y[tt], 0, self.par[2] * np.exp( 0.5 * xt ) );

    #=========================================================================
    # Define gradients of logarithm of complete data-likelihood
    #=========================================================================  
    def Dparm(self, xtt, xt, pu, pw, tt):    
        nOut = len(xtt);
        gradient = np.zeros(( nOut, self.nParInference ));
        Q1 = self.par[1]**(-1);        
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);
        R1 = self.par[2]**(-1);
        R3 = self.par[2]**(-3);       
        px = xtt - self.par[0] * xt;
        
        for v1 in range(0,self.nParInference):
            if v1 == 0:
                gradient[:,v1] = Q2 * px * xt
            elif v1 == 1:
                gradient[:,v1] = ( Q3 * px**2 - Q1 );
            elif v1 == 2:
                gradient[:,v1] = ( R3 * np.exp(-xt) * self.y[tt]**2 - R1 );
            else:
                gradient[:,v1] = 0.0;        
        return(gradient);
    
    #=========================================================================
    # Define Hessians of logarithm of complete data-likelihood
    #=========================================================================
    def DDparm(self, xtt, xt, st, at, tt):
        
        nOut = len(xtt);
        hessian = np.zeros( (nOut, self.nParInference,self.nParInference) );
        Q1 = self.par[1]**(-1);
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);
        Q4 = self.par[1]**(-4);
        R2 = self.par[2]**(-2);
        R4 = self.par[2]**(-3);     
        px = xtt - self.par[0] * xt;

        for v1 in range(0,self.nParInference):
            for v2 in range(0,self.nParInference):
                if ( (v1 == 0) & (v2 == 0) ):
                    hessian[:,v1,v2] = - xt**2 * Q2;

                elif ( (v1 == 1) & (v2 == 1) ):
                    hessian[:,v1,v2] = ( Q2 - 3.0 * Q4 * px**2 - Q1 );

                elif ( ( (v1 == 1) & (v2 == 0) ) | ( (v1 == 0) & (v2 == 1) ) ):
                    hessian[:,v1,v2] = - 2.0 * xt * Q3 * px;

                elif ( (v1 == 2) & (v2 == 2) ):
                    hessian[:,v1,v2] = ( -3.0 * R4 * np.exp(-xt) * self.y[tt]**2 + R2 );

                else:
                    hessian[:,v1,v2] = 0.0;            
        
        return(hessian);
    
    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================   
    def priorUniform(self):
        out = 1.0;
        
        if( np.abs( self.par[0] ) > 1.0 ):
            out = 0.0;
        
        if( self.par[1] < 0.0 ):
            out = 0.0;
        
        if( self.par[2] < 0.0 ):
            out = 0.0;    
        
        return( out );

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================   
    def prior(self):
        out = 0.0;
        
        # Truncated normal prior for phi (truncation by hard prior)
        if ( self.nParInference >= 1 ):
            out += normalLogPDF( self.par[0], 0.9, 0.05 );
        
        # Gamma prior for sigma
        if ( self.nParInference >= 2 ):
            out += gammaLogPDF( self.par[1], a=2.0, b=1.0/20.0 );
        
        # Gamma prior for beta
        if ( self.nParInference >= 3 ):
            out += gammaLogPDF( self.par[2], a=20.0, b=1.0/30.0 );
        
        return out;
    
    #=========================================================================
    # Define gradients of log-priors for the PMH sampler
    #=========================================================================   
    def dprior1(self,v1):
    
        if ( v1 == 0 ):
            # Truncated normal prior for phi (truncation by hard prior)
            return normalLogPDFgradient( self.par[0], 0.9, 0.05 );           
        elif ( v1 == 1):
            # Gamma prior for sigma
            return gammaLogPDFgradient( self.par[1], a=2.0, b=1.0/20.0 );
        elif ( v1 == 2):
            # Gamma prior for beta
            return gammaLogPDFgradient( self.par[2], a=20.0, b=1.0/30.0 );
        else:
            return 0.0;

    #=========================================================================
    # Define hessians of log-priors for the PMH sampler
    #=========================================================================   
    def ddprior1(self,v1,v2):
        
        if ( ( v1 == 0 ) & ( v1 == 0 ) ):
            # Truncated normal prior for phi (truncation by hard prior)
            return normalLogPDFhessian( self.par[0], 0.9, 0.05 );
        elif ( ( v1 == 1 ) & ( v1 == 1 ) ):
            # Gamma prior for sigma
            return gammaLogPDFhessian( self.par[1], a=2.0, b=1.0/20.0 );
        elif ( ( v1 == 2 ) & ( v1 == 2 ) ):
            # Gamma prior for beta
            return gammaLogPDFhessian( self.par[2], a=20.0, b=1.0/30.0 );
        else:
            return 0.0;

    #=========================================================================
    # Define hessians of log-priors for the PMH sampler
    #=========================================================================   
    def samplePrior(self):
        
        out = np.zeros( self.nParInference )
        
        # Truncated normal prior for phi (truncation by hard prior)
        if ( self.nParInference >= 1 ):
            uu = 1.2;
            while (uu > 1.0):
                uu = np.random.normal( 0.9, 0.05 );
            
            out[0] = uu;
        
        # Gamma prior for sigma
        if ( self.nParInference >= 2 ):
            out[1] = np.random.gamma( shape=2.0, scale=1.0/20.0 );
        
        # Gamma prior for beta
        if ( self.nParInference >= 3 ):
            out[2] = np.random.gamma( shape=20.0, scale=1.0/30.0 );
        
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