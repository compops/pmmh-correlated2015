import numpy       as     np
from   pmh_helpers import *
from   scipy.stats import norm
import pandas
import weave

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

    # Adaptive MCMC
    adaptHessianAfterBurnIn         = None;
    adaptHessianNoSamples           = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;
    memoryLength                    = 100;

    # Variables for constructing the file name when writing the output to file
    filePrefix                      = None;
    dataset                         = None;

    # Fixed random variables for PF
    rvnSamples                      = None;
    sigmaU                          = None;
    alpha                           = None;

    # Recursive Hessian estimate
    adaptHessianRecursively         = None;
    adaptiveU_sigmau                = None;
    adaptiveU_mean                  = None;
    adaptiveU_counter               = None;

    adaptHessianRecursivelyIterLimitFrom = None;
    adaptHessianRecursivelyIterLimitTo   = None;

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

            # Adapt Hessian
            if ( ( self.iter == ( self.nBurnIn + 1 ) ) & ( self.adaptHessianAfterBurnIn == True ) ):
                self.adaptHessian();

            if ( ( self.adaptHessianRecursively == True ) & ( self.iter < self.adaptHessianRecursivelyIterLimitTo ) ):
                self.adaptiveHessianRecursive();

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
        # Sample theta using a random walk or independent t-mixture proposal
        #=====================================================================

        if ( self.proposalTheta == 'randomWalk' ):

            if ( self.nPars == 1 ):
                self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.random.normal();
            else:
                self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

        elif ( self.proposalTheta == 'tMixture' ):
                 self.thp[self.iter,:] = self.thetaProposalTmixture();

        else:
            raise NameError("pmh: unknown proposal for theta specified.")

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
        if ( self.proposalTheta == 'randomWalk' ):
            proposalThP = 0;
            proposalTh0 = 0;

        elif ( self.proposalTheta == 'tMixture' ):
            proposalThP = 0;
            proposalTh0 = 0;
            nComp       = len( self.mixtureTproposal_prob );

            for ii in range(nComp):
                proposalThP += self.mixtureTproposal_prob[ii] * np.exp( logdmvt( self.thp[ self.iter    , :], self.mixtureTproposal_mu[0:self.nPars,ii], self.mixtureTproposal_scale[ii][0:self.nPars,0:self.nPars], self.mixtureTproposal_df[ii] ) );
                proposalTh0 += self.mixtureTproposal_prob[ii] * np.exp( logdmvt( self.th[  self.iter - 1, :], self.mixtureTproposal_mu[0:self.nPars,ii], self.mixtureTproposal_scale[ii][0:self.nPars,0:self.nPars], self.mixtureTproposal_df[ii] ) );

            proposalThP = np.log( proposalThP );
            proposalTh0 = np.log( proposalTh0 );

        else:
            raise NameError("pmh: unknown proposal for theta specified.")

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
    # Adapt the Hessian using the burn-in
    ##########################################################################
    def adaptHessian(self):
        self.invHessian = np.cov( self.th[range(self.nBurnIn-int(self.adaptHessianNoSamples),self.nBurnIn),:].transpose() );
        print('pmh: adapted Hessian using the last ' + str(self.adaptHessianNoSamples) + ' samples of the chain during burn-in with diagonal ' + str( np.round( np.diag(self.invHessian), 3 ) ) + ".");

    ##########################################################################
    # Adaptive Hessian estimate
    ##########################################################################
    def adaptiveHessianRecursive( self ):

        recursiveHessianUpdate = \
        """
        py::list ret;

        double s2, ymean1, ymean2, foo1, foo2;
        double out[2];

        s2     = previous_s2;
        ymean2 = previous_mean;

        for ( int ii = 0; ii < ( nSamples - 1 ); ++ii )
        {
            // ymean1     = ymean2;
            // ymean2     = ( ii + counter ) / ( ii + counter + 1.0 ) * ymean2 + y( ii + 1 ) / ( ii + counter + 1.0 );
            // foo1       = ymean1 * ymean1 - ymean2 * ymean2;
            // foo2       = y( ii + 1 ) * y( ii + 1 ) - s2 - ymean1 * ymean1;
            // s2         = s2 + foo1 + foo2 / ( ii + counter + 1.0 );

            s2            = ( ii + counter ) / ( ii + counter + 1.0 ) * s2 + ( y( ii + 1 ) * y( ii + 1 ) ) / ( ii + counter + 1.0 );
        }

        ret.append( s2 );
        ret.append( ymean2 );

        return_val = ret;

        """

        y             = np.asarray( self.rv ).reshape(-1);
        previous_s2   = self.adaptiveU_sigmau[ self.iter - 1];
        previous_mean = self.adaptiveU_mean;
        counter       = self.adaptiveU_counter;

        nSamples = self.rvnSamples * self.T;
        out      = weave.inline(recursiveHessianUpdate,['y','previous_s2','nSamples','previous_mean','counter'], type_converters=weave.converters.blitz, headers = ['<math.h>','<stdio.h>'], compiler = 'gcc' )

        tvar = np.min( ( np.max( ( out[0], 0.0 ) ), 1.0 ) );

        self.adaptiveU_sigmau[ self.iter ]   = tvar;
        self.adaptiveU_mean                  = out[0];
        self.adaptiveU_counter              += nSamples;
        self.sigmaU                          = tvar;

        #print( ( out[0], out[1] ) )

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

    ##########################################################################
    # Mixture of t as theta proposal
    ##########################################################################
    def thetaProposalTmixture( self ):
        nComp = len( self.mixtureTproposal_prob );

        # Sample mixture component
        idx = np.random.choice( nComp, 1, p=self.mixtureTproposal_prob )[0];

        # Sample from t distribution
        return rmvt( self.mixtureTproposal_mu[0:self.nPars,idx], self.mixtureTproposal_scale[idx][0:self.nPars,0:self.nPars], self.mixtureTproposal_df[idx] );

#############################################################################################################################
# End of file
#############################################################################################################################
