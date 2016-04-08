##############################################################################
##############################################################################
# Routines for
# Particle filtering
#
# Copyright (c) 2016 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

from smc_resampling            import *
from smc_helpers               import *
from smc_filters_correlatedRVs import *


##############################################################################
# Main class
##############################################################################

class smcSampler(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Identifier
    typeSampler      = 'smc';

    # No particles in the filter
    nPart            = None;
    
    # Threshold for ESS to resample and type of resampling scheme
    resampFactor     = None;
    resamplingType   = None;
    # resamplingType: systematic (default), multinomial, stratified

    # Initial state for the particles
    xo               = None;
    genInitialState  = None;

    sortParticles      = None;

    ##########################################################################
    # Particle filtering: wrappers for special cases
    ##########################################################################
    
    def SISrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 0;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "SIS-fixedrvs";
        self.rvpf(sys);

    # Bootstrap particle filter with fixed random variables
    def bPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF-fixedRVs";
        self.rvpf(sys);

    # Fully adapted particle filter with fixed random variables
    def faPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "faPF-fixedRVs";
        self.rvpf(sys);

    ##########################################################################
    # Particle filtering and smoothing
    ##########################################################################

    # Auxiliuary particle filter with fixed random variables
    rvpf         = proto_rvpf

    # Wrapper for trajectory reconstruction
    reconstructTrajectories = reconstructTrajectories_helper;

    # Write state estimate to file
    writeToFile = writeToFile_helper

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
