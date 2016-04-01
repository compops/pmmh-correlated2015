import numpy as np
import matplotlib.pylab as plt;
from scipy.stats import norm

def testDetailedBalance(P, pi ):
    # this calculates maxdiff for the detailed balance check.  maxdiff should be close
    #   to 0 if the detailed balance condition holds.

   maxdiff  = 0.0;

   for ii in range( len(pi) ):
      for jj in range( len(pi) ):
         d = np.abs( pi[ii] * P[ii,jj] - pi[jj] * P[jj,ii] );
         if ( d > maxdiff):
             maxdiff = d;

   return (maxdiff);

def peskunAnalysis( mu, sigmau, nGridPoints, x_lower, x_upper ):

    stepSize    = ( x_upper - x_lower ) / nGridPoints;
    evalPoints  = x_lower + stepSize * ( np.arange( nGridPoints ) - 0.5 );

    ## Evalute the target at each grid point
    pi          = np.zeros( nGridPoints )
    for ii in range(nGridPoints):
        pi[ii] = logPDFtarget( evalPoints[ii], mu )

    # Normalise the target over the grid (should look normal)
    pimax = np.max( pi )
    pi  = np.exp( pi - pimax )
    pi /= np.sum( pi )

    Pjump  = 0;

    #plt.plot(evalPoints,np.exp(pi))

    ## Construct the P for proposals to mimic MCMC
    P           = np.zeros((nGridPoints,nGridPoints))

    for ii in range(nGridPoints):
        absorbing = -1.0;

        # Probability to sample the proposal ( move from ii to jj )
        qprime = logProposal( evalPoints[ii], evalPoints[:], sigmau );

        # Compute acceptance probability ( new / old )
        # If difference is negative --> we increase the log-target and
        # we should accept, i.e. alpha = 0.0 as it is the log-acceptance probability
        foo   = logPDFtarget( evalPoints[:], mu) - logPDFtarget( evalPoints[ii] , mu ) + logProposal( evalPoints[ii], evalPoints[:], sigmau ) - logProposal( evalPoints[:], evalPoints[ii], sigmau );
        alpha = foo * ( foo < 0.0 );

        # Compute transistion probability ( q(x'|x) * alpha( x, x') * Delta )
        P[ii,:] = np.exp( qprime + alpha + np.log( stepSize ) );

        # Set diagonal element to zero
        P[ii,ii] = 0.0;

        # Normalise each row
        normFactor = np.sum( P[ii,:] );

        # Check for absorbing states
        if ( normFactor < 1e-8 ):
            print("Warning: State " + str( ii+1 ) + " is absorbing" )
            absorbing = ii;

        # Check for row that sums to more than one
        if ( normFactor > 1.001 ):
            print("Warning: P[" + str(ii+1) + "," + str(ii+1) + "] = " + str( 1.0 - normFactor ) + " < 0. Rescaling row." )
            P[ii,:] /= normFactor;
            normFactor = 1.0;

        elif ( normFactor > 1.0 ):
            normFactor = 1.0;

        # Set the diagonal element
        P[ii,ii] = 1.0 - normFactor;

        # Compute the probability of a jump (acceptance probability)
        Pjump += pi[ii] * ( 1.0 - P[ii,ii] )

        # Take care of absorbing state
        if ( absorbing > -1.0 ):
            if ( absorbing == 0 ):
                normFactor = 0.1 / nGridPoints;
                P[0,0]  = 1.0 - 1.0 * ( P[0,1] == normFactor )
                P[1,1] -= 1.0 * ( P[1,0] == normFactor * pi[0] / pi[1] );0
            else:
                print("Warning: state " + str(absorbing) + " absorbing, strange.");

    # Check detailed balance
    foo = testDetailedBalance(P, pi);

    if ( foo > 1e-4):
        print("Warning: detailed balance check 0 == " + str(foo) + ".");

    # Create the A matrix
    A           = np.zeros((nGridPoints,nGridPoints))

    for ii in range(nGridPoints):
        A[ii,:] = pi;

    # Compute the IACT
    Z = np.linalg.inv( np.diag( np.ones(nGridPoints) ) - ( P - A ) )
    B = np.diag(pi);
    foo = 2.0 * np.dot( B, Z ) - B - np.dot( B, A )

    estIACT = np.dot( np.dot( evalPoints, foo), evalPoints )

    return ( Pjump, 1.0/estIACT )

# Compute the SJD
# estSJD = 0;
#    for ii in range(nGridPoints):
#        for jj in range(ii):
#            estSJD += 2.0 * pi[ii] * P[ii,jj] * ( evalPoints[jj]  - evalPoints[ii] )**2;



def logPDFtarget( x, mu ):
    return norm.logpdf( x, mu, 1.0 );

def logProposal( old, new, sigmau ):
    return norm.logpdf( new, np.sqrt( 1.0 - sigmau**2 ) * old, sigmau );

###################################################################################
###################################################################################
###################################################################################
###################################################################################

#peskunAnalysis(0.0,1.0,750,-5.0,0.0+5.0)


muGrid    = np.arange( 0,3.75,0.25)
#muGrid    = np.arange( 0,1.5,0.25)
#muGrid    = np.arange( 1.5,2.75,0.25)
#muGrid    = np.arange( 2.75,3.75,0.25)
sigmaGrid = np.arange( 0.025, 1.025, 0.025 )
res       = np.zeros( (len(muGrid),len(sigmaGrid),4) )
bestEFF   = np.zeros( len(muGrid) )

for ii in range ( len(muGrid) ):

    for jj in range( len(sigmaGrid) ):
        res[ii,jj,0]   = muGrid[ii]
        res[ii,jj,1]   = sigmaGrid[jj]
        res[ii,jj,2:5] = peskunAnalysis(muGrid[ii],np.min((1.0,sigmaGrid[jj])),1000,-5.0,muGrid[ii]+5.0);
        print( np.round(res[ii,jj,:],3) )

    bestEFF[ii] = sigmaGrid[ np.where( res[ii,:,3] == np.max( res[ii,:,3] ) ) ];


plt.plot( muGrid, bestEFF ); plt.axis((0,4,0,1.10))


import pandas
#for ii in range(0,7):
#for ii in range(6,11):
#for ii in range(11,15):

for ii in range (len(muGrid)):
    #kk=0;
    #fileOut = pandas.DataFrame(res[kk,:,:],columns=("mu","sigmau","acceptrate","eff"));
    fileOut = pandas.DataFrame(res[ii,:,:],columns=("mu","sigmau","acceptrate","eff"));
    fileOut.to_csv('peskun-CN-' + str(ii) + '.csv');
    #kk += 1;

