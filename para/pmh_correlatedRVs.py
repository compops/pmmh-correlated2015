##############################################################################
##############################################################################
#
# correlated pmMH algorithm
#
# Copyright (c) 2016 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy       as     np
from   pmh_helpers import *
from   scipy.stats import norm
import pandas

##########################################################################
# Main class
##########################################################################

class stcPMH(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # The self.stepSize size and inverse Hessian for the sampler
    stepSize                        = None;
    invHessian                      = None;

    # How many iterations should we run the sampler for and how long is the burn-in
    nIter                           = None;
    nBurnIn                         = None;

    # When should we print a progress report? Should prior warnings be written to screen.
    nProgressReport                 = None;
    writeOutPriorWarnings           = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;

    # Variables for constructing the file name when writing the output to file
    filePrefix                      = None;
    dataset                         = None;

    # Fixed random variables for PF
    rvnSamples                      = None;
    sigmaU                          = None;
    alpha                           = None;

    # Wrappers
    calcIACT                        = calcIACT_prototype;
    calcSJD                         = calcSJD_prototype;
    calcESS                         = calculateESS_prototype;

    ##########################################################################
    # Main sampling routine
    ##########################################################################

    def runSampler(self,sm,sys,thSys,proposalType='randomWalk'):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set file prefix from model
        self.filePrefix    = thSys.filePrefix;
        self.iter          = 0;
        self.PMHtype       = 'pPMH0';
        self.PMHtypeN      = 0;
        self.nPars         = thSys.nParInference;
        self.T             = sys.T;
        self.proposalTheta = proposalType;
        self.nPart         = sm.nPart;
        self.proposeRVs    = True;

        # Initialising settings and using default if no settings provided
        setSettings(self,'pPMH0');

        # Allocate vectors
        self.ll             = np.zeros((self.nIter,1))
        self.llp            = np.zeros((self.nIter,1))
        self.th             = np.zeros((self.nIter,self.nPars))
        self.tho            = np.zeros((self.nIter,self.nPars))
        self.thp            = np.zeros((self.nIter,self.nPars))
        self.x              = np.zeros((self.nIter,self.T))
        self.xp             = np.zeros((self.nIter,self.T))
        self.aprob          = np.zeros((self.nIter,1))
        self.accept         = np.zeros((self.nIter,1))
        self.prior          = np.zeros((self.nIter,1))
        self.priorp         = np.zeros((self.nIter,1))
        self.J              = np.zeros((self.nIter,1))
        self.Jp             = np.zeros((self.nIter,1))
        self.proposalProb   = np.zeros((self.nIter,1))
        self.proposalProbP  = np.zeros((self.nIter,1))
        self.llDiff         = np.zeros((self.nIter,1))

        # Sample initial auxiliary variables (random variables)
        self.rvp    = np.random.normal( size=( self.rvnSamples, self.T ) );
        sm.rv       = self.rvp;

        # Initialise the parameters in the proposal
        thSys.storeParameters(self.initPar,sys);

        # Run the initial filter/smoother
        self.estimateLikelihood(sm,thSys);
        self.acceptParameters(thSys);

        # Inverse transform and then save the initial parameters and the prior
        self.tho[0,:] = thSys.returnParameters();
        self.prior[0] = thSys.prior()
        self.J[0]     = thSys.Jacobian();

        thSys.invTransform();
        self.th[0,:]  = thSys.returnParameters();

        #=====================================================================
        # Main MCMC-loop
        #=====================================================================
        for kk in range(1,self.nIter):

            self.iter = kk;

            # Propose parameters
            self.sampleProposal();
            thSys.storeParameters( self.thp[kk,:], sys );
            thSys.transform();

            # Calculate acceptance probability
            self.calculateAcceptanceProbability( sm, thSys );

            # Accept/reject step
            if ( np.random.random(1) < self.aprob[kk] ):
                self.acceptParameters( thSys );
            else:
                self.rejectParameters( thSys );

            # Write out progress report
            if np.remainder( kk, self.nProgressReport ) == 0:
                progressPrint( self );

            # Write out progress at some intervals
            if ( self.writeOutProgressToFile ):
                if np.remainder( kk, self.writeOutProgressToFileInterval ) == 0:
                    self.writeToFile( sm );

        progressPrint(self);
        self.thhat = np.mean( self.th[ self.nBurnIn:self.nIter, : ] , axis=0 );
        self.xhats = np.mean( self.x[ self.nBurnIn:self.nIter, : ] , axis=0 );

    ##########################################################################
    # Sample the proposal
    ##########################################################################
    def sampleProposal(self,):
        #=====================================================================
        # Sample u using a mixture of a global move and Crank-Nicholson
        #=====================================================================

        u = np.random.uniform()

        if ( u < self.alpha ):
            # Global move
            self.rvp    = np.random.normal( size=(self.rvnSamples,self.T) );
        else:
            # Local move
            self.rvp    = np.sqrt( 1.0 - self.sigmaU**2 ) * self.rv + self.sigmaU * np.random.normal( size=(self.rvnSamples,self.T) );

        #=====================================================================
        # Sample theta using a random walk
        #=====================================================================

        if ( self.nPars == 1 ):
            self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.random.normal();
        else:
            self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

    ##########################################################################
    # Calculate Acceptance Probability
    ##########################################################################
    def calculateAcceptanceProbability(self, sm,  thSys, ):

        # Check the "hard prior"
        if (thSys.priorUniform() == 0.0):
            if (self.writeOutPriorWarnings):
                print("The parameters " + str( self.thp[ self.iter,:] ) + " were proposed.");
            return None;

        # Run the smoother to get the ll-estimate, gradient and hessian-estimate
        self.estimateLikelihood(sm,thSys);

        # Compute the part in the acceptance probability related to the non-symmetric parameter proposal
        proposalThP = 0;
        proposalTh0 = 0;

        # Compute prior and Jacobian
        self.priorp[ self.iter ]    = thSys.prior();
        self.Jp[ self.iter ]        = thSys.Jacobian();

        # Compute the acceptance probability
        self.aprob[ self.iter ] = np.exp( self.llp[ self.iter, :] - self.ll[ self.iter-1, :] + proposalTh0 - proposalThP + self.priorp[ self.iter, :] - self.prior[ self.iter-1, :] + self.Jp[ self.iter, :] - self.J[ self.iter-1, :] );

        # Store the proposal calculations
        self.proposalProb[ self.iter ]  = proposalTh0;
        self.proposalProbP[ self.iter ] = proposalThP;
        self.llDiff[ self.iter ]        = self.llp[ self.iter, :] - self.ll[ self.iter-1, :];

    ##########################################################################
    # Run the SMC algorithm and get the required information
    ##########################################################################
    def estimateLikelihood(self,sm,thSys):

        # Set the auxiliary variables
        sm.rv = self.rvp;

        # Estimate the state and log-likelihood
        sm.filter(thSys);
        self.llp[ self.iter ]        = sm.ll;
        self.xp[ self.iter, : ]      = sm.xtraj;

        return None;

    ##########################################################################
    # Helper if parameters are accepted
    ##########################################################################
    def acceptParameters(self,thSys,):
        self.th[self.iter,:]        = self.thp[self.iter,:];
        self.tho[self.iter,:]       = thSys.returnParameters();
        self.x[self.iter,:]         = self.xp[self.iter,:];
        self.ll[self.iter]          = self.llp[self.iter];
        self.accept[self.iter]      = 1.0;
        self.prior[self.iter,:]     = self.priorp[self.iter,:];
        self.J[self.iter,:]         = self.Jp[self.iter,:];
        self.rv                     = np.array( self.rvp, copy=True);

    ##########################################################################
    # Helper if parameters are rejected
    ##########################################################################
    def rejectParameters(self,thSys,):
        self.th[self.iter,:]        = self.th[self.iter-1,:];
        self.tho[self.iter,:]       = self.tho[self.iter-1,:];
        self.x[self.iter,:]         = self.x[self.iter-1,:];
        self.ll[self.iter]          = self.ll[self.iter-1];
        self.prior[self.iter,:]     = self.prior[self.iter-1,:]
        self.J[self.iter,:]         = self.J[self.iter-1,:];
    
    ##########################################################################
    # Helper: compile the results and write to file
    ##########################################################################
    def writeToFile(self,sm=None,fileOutName=None):

        # Set file name from parameter
        if ( ( self.fileOutName != None ) & (fileOutName == None) ):
            fileOutName = self.fileOutName;

        # Construct the columns labels
        columnlabels = [None]*(2*self.nPars+3);
        for ii in xrange(2*self.nPars+3):  columnlabels[ii] = ii;

        for ii in range(0,self.nPars):
            columnlabels[ii]               = "th" + str(ii);
            columnlabels[ii+self.nPars]    = "thp" + str(ii);

        columnlabels[2*self.nPars]   = "acceptProb";
        columnlabels[2*self.nPars+1] = "loglikelihood";
        columnlabels[2*self.nPars+2] = "acceptflag";

        # Compile the results for output
        out = np.hstack((self.th,self.thp,self.aprob,self.ll,self.accept));

        # Write out the results to file
        fileOut = pandas.DataFrame(out,columns=columnlabels);
        if (fileOutName == None):
            if hasattr(sm, 'filterType'):
                if ( sm.filterType == "kf" ):
                    fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) + '/' + str(self.dataset) + '.csv';
                else:
                    fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) + '_N' + str(sm.nPart) + '/' + str(self.dataset) + '.csv';
            else:
                # Fallback
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '/' + str(self.dataset) + '.csv';

        ensure_dir(fileOutName);
        fileOut.to_csv(fileOutName);

        print("writeToFile: wrote results to file: " + fileOutName)
        
#############################################################################################################################
# End of file
#############################################################################################################################
