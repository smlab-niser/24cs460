import numpy as np
from scipy.special import erf

# Distribution functions
rtp = np.sqrt(2 * np.pi)
def NormDist(x,SIG=1):
    return np.exp(-x**2/(2*(SIG**2))) / (SIG * rtp)

def ExpDist(x,SIG=1):
    return np.exp(-np.abs(x)/SIG) / (2*SIG)

def TruncParam(lb,ub, SIG=1):
    """returns the phi values for TruncDist function"""
    phi1 = (1+erf(ub/ (SIG*np.sqrt(2)) ))/2
    phi2 = (1+erf(lb/ (SIG*np.sqrt(2)) ))/2
    return phi1 , phi2

def TruncNorm(x, phi1=1, phi2=0, SIG=1):
    """returns Truncated Normal distribution given the phi values from TruncParam"""
    return NormDist(x, SIG) / (phi1 - phi2)

def p1db_12(x,t):
    """returns the PDF(mod psi squared) for sum of n=1 and 2 of particle in 1d box"""
    L = 1       # Length of box
    h = 1       # hbar
    p = np.pi   # pi
    m = 1       # mass
    k = (p**2 * h) / (2* m * L**2) # k/hbar in the usual notation

    a = np.sin(p*x/L) ** 2
    b = np.sin(2*p*x/L) ** 2
    c = 2 * np.sin(p*x/L) * np.sin(2*p*x/L) * np.cos(3*k*t)
    return (a + b + c)/L

def gaussian_advection(x,t):
    """returns gaussian scalar filed advected through the medium"""
    u = 2.0     # velocity
    y = np.exp(-((x-0.0)/u))
    return y

# Dist to points
def dist2point(f,n,lb=0,ub=1,RES=100):
    """return n points sampled from f in the range lb-ub with grid resolution RES<<n"""
    #find max value of function
    xs = np.linspace(lb,ub)
    fmax = np.max(f(xs))

    # return n sample from PDF f
    samples = []
    while len(samples) < n:
        x = np.random.uniform(0,ub)
        y = np.random.uniform(0,fmax)
        if y < f(x):
            samples.append(x)
    
    return np.array(samples)

# Point to dist
def point2dist_grid(arr,lb,ub,RES=100):
    """returns the approximate PDF for given input array in grid or functional (func) form in given lower and upper bounds """
    phi1, phi2 = TruncParam(lb,ub)
    NUM = len(arr)

    amat = np.tile(arr,(RES,1))     # arr matrix
    amat = amat.T

    x = np.linspace(lb,ub,RES)
    xmat = np.tile(x,(NUM,1))

    xs = x-amat
    y = np.sum(TruncNorm(xs, phi1, phi2, 1.1),axis=0) / NUM
    # The SIG is inceased to lower the peak, compensating such that integral after PointGen is 1
    return x,y

