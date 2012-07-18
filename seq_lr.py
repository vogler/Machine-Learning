#!/usr/bin/env python
"""
__author__: Martin Felder
Demo for sequential bayesin learning
for linear regression models. See
Chapter 3.3 of 'Pattern recognition 
and machine learning' by Chris Bishop, 2001
"""


import scipy as sp;
import numpy as npy;
from numpy import mat
from pylab import *
from copy import copy


def bivar_gauss(u, v, m, Sinv):
    x = array([u,v])
    return exp(-0.5*dot(dot((x-m).transpose(),Sinv),x-m))


def mesh2vector(U, V):
    """convert given mesh to a vector with v[:,i] giving gridpoint i's coords"""
    return mat([U.flatten(),V.flatten()])


def vector2mesh(vec, template):
    """convert given mesh to a vector with v[:,i] giving gridpoint i's coords"""
    return vec.reshape(size(template,axis=0),size(template,axis=1))


def multivar_gauss(X, m, Sinv):
    """computes a multivariate Gaussian for multiple sample vectors (cols of X)"""
    nVec = size(X,axis=1)
    expo = empty(nVec)
    for i in range(nVec):
        z = X[:,i]-m
        expo[i] = z.getT()*Sinv*z
    # normalize
    g = exp(-0.5*expo)
    return g/g.sum()


class basisFunctionAffine:

    def getNoisyVal(self,*args):
        """0=a0, 1=a1, 2,3=random range, 4=#points, 5=noise"""
        nPoints = args[4]
        x = npy.random.uniform(args[2],args[3],nPoints)
        y = self.dotProd(x,args[0],args[1]) + npy.random.normal(0,0.2,nPoints)
        return x, y

    def dotProd(self,x,*args):
        """returns dot product of basis function with given vector"""
        return args[0] + args[1]*x

    def designMatrix(self,X):
        """returns design matrix of basis function"""
        return npy.mat([ones(len(X)),X]).transpose()


class bayesLinReg:

    def __init__(self, alpha, beta):
        """inititalize with a Gaussian prior and no measurements"""
        self.alpha = alpha
        self.beta  = beta
        self.initPrior()
        self.setBasisFct('basisFunctionAffine')

    def initPrior(self):
        self.pMean = mat(zeros([2,1]))
        self.pCov = mat(identity(2) / self.alpha)

    def prior(self,U,V):
        if self.pCov[0,1] != self.pCov[1,0]:
            print "Error: Asymmetric prior covariance!"
            sys.exit(1)
        return vector2mesh(multivar_gauss(mesh2vector(U,V), self.pMean, self.pCov.getI()), U)
        #bivariate_normal(U, V, sigmax=sqrt(self.pCov[0,0]), sigmay=sqrt(self.pCov[1,1]), mux=self.pMean[0], muy=self.pMean[1], sigmaxy=sqrt(self.pCov[0,1]))

    def likelihood(self,x,t,U,V):
        """likelihood function for a single observation"""
        #K = len(U)
        #lk = npy.empty_like(V)
        std = 1.0/self.beta
        lk = normpdf(t, self.basisFct.dotProd(x,U,V), std)
        return lk

    def setBasisFct(self, fctname):
        self.basisFct = eval(fctname+'()')

    def getPosteriorParams(self,X,T):
        """return posterior mean and inverse(!) covariance depending on samples given (Bishop eqn. 3.50/51)"""
        phi = self.basisFct.designMatrix(X)
        SnI = self.pCov.getI() + self.beta*(phi.getT()*phi)
        mn  = SnI.getI()*(self.pCov.getI()*self.pMean + self.beta*phi.getT()*mat(T).getT())
        return mn, SnI

    def posterior(self,X,T,U,V):
        """return posterior depending on samples given (Bishop eqn. 3.49)"""
        mn, SnI = self.getPosteriorParams(X,T)
        return vector2mesh(multivar_gauss(mesh2vector(U,V), mn, SnI), U)
        #return npy.nan_to_num(bivariate_normal(U, V, sigmax=sqrt(Sn[0,0]), sigmay=sqrt(Sn[1,1]), mux=mn[0], muy=mn[1], sigmaxy=sqrt(Sn[0,1])))
        #return normpdf(U,mn[0],Sn[0,0])*normpdf(V,mn[1],Sn[1,1])

    def drawW(self,X,T):
        """create a design matrix from given samples"""
        mn, SnI = self.getPosteriorParams(X,T)
        return npy.random.multivariate_normal(array(mn).flatten(), SnI.getI())

    
noise = 0.2            # stddev of noise in the data
beta = (1.0/noise)**2  # precision parameter of data
alpha = 2.0            # precision of prior
trueW = array([-0.3,0.5])

# generate the data
f = basisFunctionAffine()
X = []
Y = copy(X)
x1 = array([-1,1])  # for plotting

# prepare the prior/posterior density
du, dv = 0.02, 0.02
u = arange(-1.0, 1.0, du)
v = arange(-1.0, 1.0, dv)
U,V = meshgrid(u, v)

# setup Bayesian regression
bayes = bayesLinReg(alpha, beta)
bayes.setBasisFct('basisFunctionAffine')

# setup  graphics
fig = figure(figsize=[18,5])
pLike = fig.add_subplot(131)
pLike.set_title('likelihood', fontsize=20)
pPost = fig.add_subplot(132)
pPost.set_title('prior density', fontsize=20)
pPost.set_xlabel('$w_0$', fontsize=15)
pPost.set_ylabel('$w_1$', fontsize=15)

pDat = fig.add_subplot(133)
pDat.set_title(r'data space', fontsize=20)

nP = 0
# plot the prior density
p = bayes.prior(U, V)
pPost.imshow(p, interpolation='bilinear', origin='lower',\
             extent=(-1,1,-1,1))
# plot true data
pDat.plot(x1, f.dotProd(x1,trueW[0],trueW[1]), color='g',alpha=1.0,linewidth=3)
pDat.hold(True)
# plot the data
for i in range(6):
    w1 = npy.random.uniform(-1,1)
    w2 = npy.random.uniform(-1,1)
    pDat.plot(x1, f.dotProd(x1,w1,w2), color='r',alpha=0.5,linewidth=3)
pDat.set_ylim(-1,1)
pDat.set_xlabel('$x$', fontsize=15)
pDat.set_ylabel('$t$', fontsize=15)
pDat.hold(False)

class InteractiveBayes(object):
    def onclick(self, event):
    
        if not event.button==1: return True
    
        # add the click locations
        X.append(event.xdata)
        Y.append(event.ydata)
        nP = len(X)
        print nP, event.xdata, event.ydata
        
        # plot the likelihood
        p = bayes.likelihood(X[nP-1],Y[nP-1],U,V)
        im = pLike.imshow(p, interpolation='bilinear', origin='lower',\
                    extent=(-1,1,-1,1))
        pLike.hold(True)
        pLike.plot([trueW[0]],[trueW[1]],'+',markersize=10,markeredgecolor='w',markeredgewidth=2)
        pLike.set_xlabel('$w_0$', fontsize=15)
        pLike.set_ylabel('$w_1$', fontsize=15)
        hold(False)
    
        # plot the posterior density
        p = bayes.posterior(X[0:nP],Y[0:nP], U, V)
        im = pPost.imshow(p, interpolation='bilinear', origin='lower',\
                    extent=(-1,1,-1,1))
        pPost.hold(True)
        pPost.plot([trueW[0]],[trueW[1]],'+',markersize=10,markeredgecolor='w',markeredgewidth=2)
        pPost.set_xlabel('$w_0$', fontsize=15)
        pPost.set_ylabel('$w_1$', fontsize=15)
        pPost.set_title('posterior density', fontsize=20)
        hold(False)        
    
        # plot true data
        pDat.plot(x1, f.dotProd(x1,trueW[0],trueW[1]), color='g',alpha=1.0,linewidth=3)
        pDat.hold(True)
        
        # plot the data
        for i in range(6):
            w1, w2 = bayes.drawW(X[0:nP],Y[0:nP])
            pDat.plot(x1, f.dotProd(x1,w1,w2), color='r',alpha=0.5,linewidth=3)
        pDat.plot(X[0:nP],Y[0:nP],'o',markeredgecolor='b', markerfacecolor='w',markeredgewidth=1.5)
        #plot(u,p.sum(axis=0)*10-1,linewidth=2,color='g')  # plot marginalized gaussian
        pDat.set_ylim(-1,1)
        pDat.set_title(r'data space', fontsize=20)
        pDat.set_xlabel('$x$')
        pDat.set_ylabel('$t$')
        pDat.hold(False)
        fig.canvas.draw()
        
bay = InteractiveBayes()
fig.canvas.mpl_connect('button_press_event', bay.onclick)
show()
