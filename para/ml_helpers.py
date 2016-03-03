##############################################################################
##############################################################################
# Default settings and helpers for
# Maximum-likelihood inference
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy as np
import pandas
import os

##############################################################################
# Set default settings if needed
##############################################################################
def setSettings(ml,vers):

    if ( ml.initPar == None ):
        raise NameError("ml-opt (initPar): no initial parameters given, exiting...");

    if ( ml.dataset == None ):
        ml.dataset = 0;
        print("ml-opt (dataset): no number of data set given, defaulting to " + str(ml.dataset) + ".");

    if ( ml.filePrefix == None ):
        ml.filePrefix = "model";
        print("ml-opt (filePrefix): no short name for model given, defaulting to " + str(ml.filePrefix) + ".");

    #=====================================================================
    # Settings for the BFGS using the smooth particle filter
    #=====================================================================
    if ( vers == "smoothOpt" ):

        if ( ml.parBounds == None ):
            raise NameError("ml-opt-smooth (parBounds): no parameter bounds (parBounds) given.");

    #=====================================================================
    # Settings for the EM algorithm
    #=====================================================================
    if ( vers == "em" ):

        if ( ml.verbose == None ):
            print("em (verbose): defaulting to verbose algorithm.");
            ml.verbose = True;

        if ( ml.tolLevel == None ):
            ml.tolLevel = 1e-6;
            print("em (tolLevel): defaulting to " + str(ml.tolLevel) + " as tolerance level.");

        if ( ml.noisyTolLevel == None ):
            ml.noisyTolLevel = 5;
            print("em (noisyTolLevel): defaulting to checking tolerance level over the last " + str(ml.noisyTolLevel) + " iterations.");

        if ( ml.maxIter == None ):
            ml.maxIter = 200;
            print("em (maxIter): defaulting to " + str(ml.maxIter) + " as maximum no iterations.");

        if ( ( ml.initCondPath == None ) & ( ml.optMethod == "psaem" ) ):
            ml.initCondPath = np.zeros( (1, ml.T) );
            print("em (initCondPath): defaulting to having zero as initial trajectory for cPF.");

        if ( ml.stochasticApproximation == False ):
            ml.stochasticApproximationFrom = 0;
            ml.alpha                       = 2.0/3.0;

        if ( ml.stochasticApproximation == None ):
            ml.stochasticApproximation     = True;
            ml.stochasticApproximationFrom = 0;
            ml.alpha                       = 2.0/3.0;
            print("em (stochasticApproximation,stochasticApproximationFrom,alpha): defaulting to use stochastic approximation from iteration kk:" + str(ml.stochasticApproximationFrom) + " using kk**(-a) with a: " + str() + ".");
        
        if ( ( ml.writeOutProgressToFileInterval == None ) & ( ml.writeOutProgressToFile == None ) ):
            ml.writeOutProgressToFileInterval  = 50;
            ml.writeOutProgressToFile          = True;
            print("ml (writeOutProgressToFileInterval,writeOutProgressToFile): defaulting write out current progress to file for each: " + str(ml.writeOutProgressToFileInterval) + " step of the algorithm.");

    #=====================================================================
    # Settings for the gradient and Newton algorithms
    #=====================================================================
    if ( vers == "simpleOpt" ):

        if ( ml.stepSize != None ):
            ml.step          = ml.stepSize;
            ml.adaptStepSize = False;

        if ( ml.verbose == None ):
            print("ml-opt (verbose): defaulting to verbose algorithm.");
            ml.verbose = True;

        if ( ml.tolLevel == None ):
            ml.tolLevel = 1e-6;
            print("ml-opt (tolLevel): defaulting to " + str(ml.tolLevel) + " as tolerance level.");

        if ( ml.noisyTolLevel == None ):
            ml.noisyTolLevel = 5;
            print("ml-opt (noisyTolLevel): defaulting to checking tolerance level over the last " + str(ml.noisyTolLevel) + " iterations.");

        if ( ml.maxIter == None ):
            ml.maxIter = 100;
            print("ml-opt (maxIter): defaulting to " + str(ml.maxIter) + " as maximum no iterations.");

        if ( ml.adaptStepSize == None ):
            ml.adaptStepSize     = True;
            ml.adaptStepSizeFrom = 0;
            ml.alpha             = 2.0/3.0;
            ml.gamma             = 0.001;
            print("ml-opt (adaptStepSize,adaptStepSizeFrom,alpha,gamma): defaulting to adapting step sizes staring from iteration kk:" + str(ml.adaptStepSizeFrom) + "using g * kk**(-a) with a: " + str() + " and g: " + str() +".");

    if ( ( vers == "noisyQuasiNewtonOpt" ) | ( vers == "noisyLimitedMemoryQuasiNewtonOpt" ) ):

        if ( ml.maxIter == None ):
            ml.maxIter = 100;
            print("ml-opt (maxIter): defaulting to " + str(ml.maxIter) + " as maximum no iterations.");

        if ( ml.verbose == None ):
            print("ml-opt (verbose): defaulting to verbose algorithm.");
            ml.verbose = True;

        if ( ml.tolLevel == None ):
            ml.tolLevel = 1e-6;
            print("ml-opt (tolLevel): defaulting to " + str(ml.tolLevel) + " as tolerance level.");

        if ( ml.epsilon == None ):
            ml.epsilon = 1e-4;
            print("ml-opt (epsilon): defaulting to " + str(ml.epsilon) + " initial diagonal estimate of the Hessian.");

        if ( ml.cc == None ):
            ml.cc = 0.1;
            print("ml-opt (cc): defaulting to " + str(ml.cc) + " scaling of the new contribution to the Hessian estimate.");

        if ( ml.tau == None ):
            ml.tau = 100.0;
            print("ml-opt (tau): defaulting to " + str(ml.tau) + " as tau in the adaptive step size tau/(tau+kk)*eta.");

        if ( ml.eta == None ):
            ml.eta = 0.05;
            print("ml-opt (eta): defaulting to " + str(ml.eta) + " as eta in the adaptive step size tau/(tau+kk)*eta.");

        if ( ml.lam == None ):
            ml.lam = 0.1;
            print("ml-opt (lam): defaulting to " + str(ml.lam) + " as the regularisation factor.");

    if ( ( vers == "noisyLimitedMemoryQuasiNewtonOpt" ) ):

        if ( ml.memoryLength == None ):
            ml.memoryLength = 10;
            print("ml-opt (memoryLength): defaulting to " + str(ml.memoryLength) + " length of memory in L-BFGS.");

    #=====================================================================
    # Settings for the SPSA algorithm
    #=====================================================================
    if ( vers == "spsa" ):

        if ( ml.verbose == None ):
            print("ml-spsa (verbose): defaulting to verbose algorithm.");
            ml.verbose = True;

        if ( ml.tolLevel == None ):
            ml.tolLevel = 1e-6;
            print("ml-spsa (tolLevel): defaulting to " + str(ml.tolLevel) + " as tolerance level.");

        if ( ml.noisyTolLevel == None ):
            ml.noisyTolLevel = 5;
            print("ml-spsa (noisyTolLevel): defaulting to checking tolerance level over the last " + str(ml.noisyTolLevel) + " iterations.");

        if ( ml.maxIter == None ):
            ml.maxIter = 100;
            print("ml-spsa (maxIter): defaulting to " + str(ml.maxIter) + " as maximum no iterations.");

        if ( ml.alpha == None ):
            ml.alpha = 0.602;
            print("ml-spsa (alpha): defaulting to alpha: " + str(ml.alpha) + " as suggested in Spall(1998).");

        if ( ml.gamma == None ):
            ml.gamma = 0.101;
            print("ml-spsa (gamma): defaulting to gamma: " + str(ml.gamma) + " as suggested in Spall(1998).");

        if ( ml.A == None ):
            ml.A = 0.10 * ml.maxIter;
            print("ml-spsa (A): defaulting to A: " + str(ml.A) + " as 10% of maxIter as suggested in Spall(1998).");

    #=====================================================================
    # Settings for the GPO algorithm
    #=====================================================================
    if ( vers == "gpo" ):
        if ( ml.maxIter == None ):
            ml.maxIter = 100;
            print("gpo (maxIter): defaulting to " + str(ml.maxIter) + " as maximum no iterations.");

        if ( ml.verbose == None ):
            print("gpo (verbose): defaulting to verbose algorithm.");
            ml.verbose = True;

        if ( ml.tolLevel == None ):
            ml.tolLevel = 1e-3;
            print("gpo (tolLevel): defaulting to " + str(ml.tolLevel) + " as tolerance level for AQ.");

        if (  ml.epsilon == None ):
            ml.epsilon = 1e-2;
            print("gpo (epsilon): defaulting to " + str(ml.epsilon) + " as epsilon.");

        if (  ml.preIter == None ):
            ml.preIter = 50;
            print("gpo (preIter): defaulting to " + str(ml.preIter) + " pre-iterations to estimate hyperparameters.");

        if (  ml.upperBounds == None ):
            raise NameError("gpo (upperBounds): no upper parameter bounds (upperBounds) given.");

        if (  ml.lowerBounds == None ):
            raise NameError("gpo (lowerBounds): no lower parameter bounds (lowerBounds) given.");

        if (  ml.EstimateHyperparametersInterval == None ):
            ml.EstimateHyperparametersInterval = 10000;
            print("gpo (EstimateHyperparametersInterval): defaulting to not updating hyperparameters for pre-iterations.");

        if (  ml.AQfunction == None ):
            ml.AQfunction = ml.aq_ei;
            print("gpo (AQfunction): defaulting using expected improvement (EI) as aquisition function.");

        if ( ml.EstimateThHatEveryIteration == None ):
            ml.EstimateThHatEveryIteration = True;
            print("gpo (EstimateThHatEveryIteration): defaulting to estimate parameters at every iteration (set EstimateThHatEveryIteration to FALSE for speedup).");

        if ( ml.jitterParameters == None ):
            ml.jitterParameters = True;
            print("gpo (jitterParameters): defaulting to jitter parameters.");

        if ( ( ml.jitteringCovariance == None ) & ( ml.jitterParameters == True ) ):
            tmp = 0.01;
            ml.jitteringCovariance = tmp * np.diag(np.ones(ml.nPars));
            print("gpo (jitteringCovariance): defaulting to jitter parameters with Gaussian noise with variance " + str(tmp) + ".");

        if ( ml.preSamplingMethod == None ):
            ml.preSamplingMethod = "latinHyperCube";
            print("gpo (preSamplingMethod): defaulting to: latinHyperCube during the pre-iterations (alternatives: sobol, uniform).");

##########################################################################
# Helper: compile the results and write to file
##########################################################################
def writeToFile_helper(ml,sm=None,fileOutName=None,noLLests=False):
    
    # Set file name from parameter
    if ( ( ml.fileOutName != None ) & (fileOutName == None) ):
        fileOutName = ml.fileOutName;
    
    # Construct the columns labels
    if ( noLLests ):
        columnlabels = [None]*(ml.nPars+1);
    else:
        columnlabels = [None]*(ml.nPars+3);

    for ii in range(0,ml.nPars):
        columnlabels[ii]   = "th" + str(ii);

    columnlabels[ml.nPars] = "step";

    if ( noLLests == False ):
        columnlabels[ml.nPars+1] = "diffLogLikelihood";
        columnlabels[ml.nPars+2] = "logLikelihood";

    # Compile the results for output
    if ( noLLests ):
        out = np.hstack((ml.th,ml.step));
    else:
        out = np.hstack((ml.th,ml.step,ml.llDiff,ml.ll));

    # Write out the results to file
    fileOut = pandas.DataFrame(out,columns=columnlabels);

    if ( fileOutName == None ):
        if ( sm.filterType == "sPF" ):
            fileOutName = 'results/' + str(ml.filePrefix) + '/' + str(ml.optMethod) + '_' + sm.filterType + '_N' + str(sm.nPart)  + '/' + str(ml.dataset) + '.csv';
        elif ( sm.filterType == "kf" ):
            fileOutName = 'results/' + str(ml.filePrefix) + '/' + str(ml.optMethod) + '_' + sm.filterType + '/' + str(ml.dataset) + '.csv';
        else:
            fileOutName = 'results/' + str(ml.filePrefix) + '/' + str(ml.optMethod) + '_' + sm.filterType + '_' + sm.smootherType + '_N' + str(sm.nPart)  + '/' + str(ml.dataset) + '.csv';

    ensure_dir(fileOutName);
    fileOut.to_csv(fileOutName);

    print("writeToFile_helper: wrote results to file: " + fileOutName)

##############################################################################
# Calculate the pdf of a univariate Gaussian
##############################################################################
def uninormpdf(x,mu,sigma):
    return 1.0/np.sqrt( 2.0 * np.pi * sigma**2 ) * np.exp( - 0.5 * (x-mu)**2 * sigma**(-2) );

##############################################################################
# Calculate the log-pdf of a univariate Gaussian
##############################################################################
def loguninormpdf(x,mu,sigma):
    return -0.5 * np.log( 2.0 * np.pi * sigma**2) - 0.5 * (x-mu)**2 * sigma**(-2);

##############################################################################
# Calculate the log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
##############################################################################
def lognormpdf(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu

    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+numerator)

##############################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
##############################################################################
def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

##############################################################################
# Calculate vector Inf-norm
##############################################################################
def vecnorm( x ):
    return np.amax(np.abs(x))

##############################################################################
# Check if dirs for outputs exists, otherwise create them
##############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################