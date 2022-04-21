import scipy.stats as st
import numpy as np
import math
import time
import seaborn as sns
from IPython.display import Image
from random import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NO_OF_ITERATIONS=10000
def p_x1_given_x2(x,mean,cov):    #conditional probability x1/x2
    x2=x
    u1=mean[0]
    u2=mean[1]
    u=u1+(x2-u2)*cov[0][1]/cov[1][1]
    std=(cov[0][0]*cov[1][1]-cov[0][1]*cov[1][0])/cov[1][1]
    return np.random.normal(u,std)

def p_x2_given_x1(x,mean,cov):   #conditional probability x2/x1
    x1=x
    u1=mean[0]
    u2=mean[1]
    u=u2+(x1-u1)*cov[0][1]/cov[0][0]
    std=(cov[0][0]*cov[1][1]-cov[0][1]*cov[1][0])/cov[0][0]
    return np.random.normal(u,std)

#some variables
mean=0
cov=0;
x1_list=[]
x2_list=[]   

def gibbs_sampling(i): #Implement the Gibbs sampler based on the #conditional distributions. (10000 iteration)

    global x1_list,x2_list
    if len(x2_list)==0:
      x2=np.random.rand()*10
    else:
      x2=x2_list[len(x2_list)-1]
    global mean,cov
    sigma=cov
    u=mean
    x1 = p_x1_given_x2(x2,u,sigma)
    x2 = p_x2_given_x1(x1,u,sigma)
    x1_list.append(x1)
    x2_list.append(x2)
    
    plt.cla()
    plt.plot(x1_list,x2_list,'o',ms=0.8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Two-dimensional space plotting both x1 & x2")
    plt.plot([x1_list[i],x1_list[i-1]],[x2_list[i],x2_list[i-1]])
    #plt.savefig("x1_vs_X2_plot")
    

###INPUT##
a=input("Enter a:")
mean = np.array([1,2])
cov = np.array([[1,float(a)],[float(a),1]])

###Making the plot of simulations in the two-dimensional space plotting both x1 & x2,along with drawing a line between each iteration (b/w latest added) ###
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.unicode_minus'] = False

ani=FuncAnimation(plt.gcf(),gibbs_sampling,frames=NO_OF_ITERATIONS,repeat=False,interval=60)   #for annimation, it will call gibbs sampler function 
											# to the #frames times
plt.tight_layout()
#plt.savefig("x1_vs_X2_plot")
plt.show()


samples=np.zeros((len(x1_list), 2))  #samples
for i in range(len(x1_list)):
	samples[i,:]=[x1_list[i],x2_list[i]]


#### DISPLAYING THE BIVARIATE GAUSSIAN DISTRIBUTINO TRACE PLOT OF X2 WITH RESPECT TO ITERATIONS ####
plt.rcParams['figure.figsize'] = [20,8]
plt.plot([i for i in range(len(samples))],samples[:,1],linewidth = '0.2')  #trace plot of Gaussian distribution of N2(μ, Σ) with iteration (of x2)
plt.title("Trace plot of Gaussian distribution of N2(μ, Σ) with iteration (of x2)")
plt.ylabel("x2")
plt.xlabel("Iteration")
#plt.savefig("traceplot_x2")
plt.show()

#### DISPLAYING THE BIVARIATE GAUSSIAN DISTRIBUTINO TRACE PLOT OF X1 WITH RESPECT TO ITERATIONS ####
plt.rcParams['figure.figsize'] = [20,8.1]
plt.plot([i for i in range(len(samples))],samples[:,0],linewidth = '0.2')  #trace plot of Gaussian distribution of  N2(μ, Σ) with iteration (of x1)
plt.ylabel("x1")
plt.xlabel("Iteration")
plt.title("Trace plot of Gaussian distribution of  N2(μ, Σ) with iteration (of x1)")
#plt.savefig("traceplot_x1")
plt.show()

#### DISPLAYING FINAL GRAPH OF X1 VS X2 ####
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
ax.plot(samples[:,0],samples[:,1], 'o',ms=0.8)
ax.set_title('Final Two-dimensional space plotting both x1 & x2')
plt.xlabel("x1")
plt.ylabel("x2")
#plt.savefig("Final_x1_vs_X2_plot")
plt.show()
plt.tight_layout()
