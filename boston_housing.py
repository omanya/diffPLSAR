# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:54:52 2017

@author: Maria Osipenko

On the estimation of Partially linear spatial autoregressive models 
   on the example of the well known Boston Housing dataset.

Reference for Partially linear spatial autoregressive models: 
    Preprint: Difference-based estimation of partially linear
    spatial autoregressive models
    
References for the dataset and some estimation approaches:
    Harrison, David, and Daniel L. Rubinfeld, 
    Hedonic Housing Prices and the Demand for Clean Air, 
    Journal of Environmental Economics and Management, 
    Volume 5, (1978), 81-102. (Original data)

    Gilley, O.W., and R. Kelley Pace, On the Harrison and Rubinfeld Data, 
    Journal of Environmental Economics and Management, 31 (1996), 403-405. 
    (Provided corrections and examined censoring)

    Pace, R. Kelley, and O.W. Gilley, 
    Using the Spatial Configuration of the Data to Improve Estimation, 
    Journal of the Real Estate Finance and Economics, 14 (1997), 333-340.
    
Summary:
    We propose a simple difference-based approach to estimation of
    partially linear spatial autoregressive models with general error struc-
    ture: y = m(t)+ Xb + l W y + e
    y univariate response, m() smooth fucntion, t explanatory vars entering m,
    X explanatory vars with linear effects, W spatial weight matrix 
    with zero diagonal.
        
    The estimation contains several steps: 
        
        - the smooth non-linear part of the model ist removed by differencing. 
          
        - the linear parameters are estimated from the differenced data using 
          two-stage least squares as for purely linear spatial autoregression 
          (SAR). Standart SAR inference can be performed including the 
          construction of the spatial autocorrelation consistent standart errors 
          for the estimated parameters.
          
        - the nonlinear part ist estimated from the residuals of SAR using some
          nonparametric technique. Here we use Nadaraya-Watson regression.
    
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
#%%
#load the corrected boston housing dataset
#from: 
boston = pd.read_csv("C:\\Users\\Mascha\\Documents\\boston.csv",header=0)

#take a look at the data
list(boston)
boston.head()
boston.shape

# Transformations fo features (taken from Pace, Kelley and Gilley (1997))
feats_to_log = ["CMEDV","RAD","DIS","LSTAT"]
feats_to_sqr = ["NOX","RM"]
#feats_to_drop = ["ZN","CHAS","INDUS"]
boston[feats_to_log] = np.log(boston[feats_to_log])
boston[feats_to_sqr] = boston[feats_to_sqr]**2
#boston.drop(feats_to_drop,axis = 1, inplace = True)

#%%
""" We are going to estimate a partial linear model: 
    CMEDV = m(LAT,LON)+ Xb + lambda W CMEDV + error
    m() is a smooth function nonparametrized
    Xb + lambda W CMEDV is the parametric part, X contains exploratory variables
    lambda W CMEDV is the spatial autoregressive part with lambda spatial autoregressive parameter
    W is the spatial weight matrix with zero diagonal reflecting the spatial relationship in CMEDV

    There are several technical assumptions on the model parts, see Preprint.
"""
# Variable definitions
x,y,coord = boston.loc[:,"CRIM":"LSTAT"],boston["CMEDV"],boston.loc[:,"LON":"LAT"]
#x contains the explanatory variables of the parametric part
#y contains the dependent variable
#coord contains the explanatory variables of the nonparametric part
n = x.shape[0]
p = x.shape[1]
#demean the dependent var
y = y - y.mean()

#%%
from sklearn.neighbors import NearestNeighbors

coord_a = np.asarray(coord)
#fit the Nearestneighbors
nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(coord_a)
distances, indices = nbrs.kneighbors(coord_a)
##find the optimal path with minimum sum of distances
path = indices[np.argmin(distances.sum(axis =1),axis=0)]
np.savetxt("path.csv",path)

#%% 
#Rearange the data
coord_a = coord_a[path]
x_a = np.asarray(x)[path]
y_a = np.asarray(y)[path]

#Define the difference matrix D for the differencing order r=4 (the optimized weights are taken from Hall et al. (1990))
dv = [0.2708,-0.0142,0.6909,-0.4858,-0.4617]   
d = np.zeros((n,n))
r = len(dv)
for i in range(n-r):
    d[i][i:(r+i)] = dv

#%%
#Define the spatial weight matrix W
from scipy.spatial.distance import pdist, squareform

#define distance and the bandwidth behind which there is no spatial dependence
dn = 1#np.floor(n**(1/4)) # the distance bandwidth (rule of thumb)
w = squareform(pdist(coord_a,'seuclidean'))

#normalize the weight matrix
from sklearn.preprocessing import normalize

wn =np.copy(w)
wn[w<=dn] = 1
wn[w>dn] = 0
#wn[np.diag_indices_from(wn)] = 0. # Weight matrix should have zeros on the diagonal
wn = normalize(wn, axis=1, norm='l1')

#%%
#Define Wy and intstruments H
wy = np.dot(wn,y_a).reshape((x_a.shape[0],1))
h = np.concatenate((x_a,np.dot(wn,x_a),np.dot(wn,np.dot(wn,x_a))),axis = 1)
z = np.concatenate((x_a,wy),axis = 1)

#Do the differencing by premultiplying with D
dx = np.dot(d,x_a)
dy = np.dot(d,y_a)
dwy = np.dot(d,wy)
dh = np.dot(d,h)

#%%
#Do the estimation by two-stage-least squares (2SLS)

#first stage 2SLS
from numpy.linalg import inv

proj=np.dot(dh,np.dot(inv(np.dot(dh.T,dh)),dh.T))
dwyh = np.dot(proj,dwy)
dz = np.concatenate((dx, dwy),axis = 1)
dzh = np.concatenate((dx, dwyh),axis = 1)

#%%

#second stage 2SLS
#fit linear model to the differenced data
lm_fit = sm.OLS(dy,dzh).fit()
print(lm_fit.summary())

#%%
#compute the spatial autocorrelation consistent covariance matrix estimator (SHAC)

# Compute the differenced model residuals 
du = lm_fit.resid
def kern(x,v,dn):
    """ compute the kernel weight for x using a triangular kernel function
    """
    
    K = np.zeros(x.shape)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]<1:
                K[i][j] = (1-x[i][j]/dn)**v
        
    return(K)
# Compute kernel weights       
k = kern(w,2,dn)
#Phi
phi=np.multiply(np.dot(du,du.T),k)
#SHAC estimator
shac=1/n*np.dot(inv(np.dot(dzh.T,dz)),np.dot(np.dot(dzh.T,np.dot(phi,dzh)),inv(np.dot(dzh.T,dz))))

#print the results
results = pd.DataFrame({"Coefficients": lm_fit.params, 
                    "standart errors": lm_fit.bse,
                    "spatial standart errors": np.sqrt(np.diag(shac))})#se under spatial dependence
names = list(x)
names.append("lambda (spatial param)")
results.index = names
print(results)

#%%
#compute (non-differenced) residuals
resid = y_a - z.dot(lm_fit.params) # apply the estimated coefficients to NON-DIFFERENCED data

#fit nonparametric part: Nadaraya-Watson kernel regression (local constant regression)
from statsmodels.nonparametric.kernel_regression import KernelReg

llreg = KernelReg(resid,coord_a,"cc","lc","aic") #continuos variables, local constant, choose bandwidth by AIC
#test statistical significance of the vars in the nonparametric regression
print('Significance of the nonparametric regressors',llreg.sig_test([0,1])) # 99% confidence level
#%%
#plot resid vs lat,lon
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

gn = 100 #grid intervals
gy= np.linspace(coord_a[:,0].min(),coord_a[:,0].max(),gn)
gx= np.linspace(coord_a[:,1].min(),coord_a[:,1].max(),gn)
gxx, gyy = np.meshgrid(gx,gy)

# compute predictions for the grid
condmean,_ = llreg.fit(np.concatenate((gyy.reshape((gn**2,1)),
                                       gxx.reshape((gn**2,1))),axis=1))
condmean = condmean.reshape((gn,gn)).T

# 1. Contour plot
plt.figure()
cnt = plt.contour(gx, gy, condmean)
#cnt = plt.contour(gx, gy, condmean,colors='k')#no color
plt.clabel(cnt, inline=1, fontsize=10)
#plt.savefig('contour.pdf')
plt.title('Nadaraya-Watson estimate of the location effect')

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# 2. Surface
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface
surf = ax.plot_surface(gy,gx,condmean, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the axes
#z
ax.set_zlim(1, 1.5)
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
#x
ax.set_xlim(-71.5, -70.5)
ax.xaxis.set_major_locator(LinearLocator(6))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
#y
ax.set_ylim(41.8, 42.3)
ax.yaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
