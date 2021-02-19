# ------------------------------------------------------------------------- #
#  File: project1.py                                                        #
#  Author: Christopher Flowers                                              # 
#  Date: 2012-02-19                                                         #
#  Purpose: To solve the Lane-Emden equation numerically via RK2 and plot.  #
# ------------------------------------------------------------------------- #

# -------------------- #
#     Dependencies     #
# -------------------- #

import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------- #
# Solution Parameters  #
# -------------------- #

h = 0.01                 # Step size
indexes = [1.5,3,3.25]   # Polytropic indexes to solve numerically
ps_end = 1               # Where to end power series, it is set to 1 since that is where the term stops going to infinity

ps = lambda zeta,n : 1 - (1/6)*zeta**2 + (n/120)*zeta**4     # Lambda function for theta power series
dps = lambda zeta,n : -(1/3)*zeta + (n/30)*zeta**3           # Lambda function for derivative of power series (to obtain dθ/dt for initial value of numerical solution)

# --------------------------------------- #
# Initial Conditions and Solution Ranges  #
# --------------------------------------- #

dtheta_ps = dps(ps_end,1)  # Initial value for numerical portion of dθ/dt
zeta = np.arange(ps_end+h,10+h,h)  # Numerical portion of zeta
theta = [[]]*3                     # Multi-dimensional array to store each solution of theta.

# --------- #
# Functions #
# --------- #

def func2(zeta,x,n):       # State function
    theta,v = [x[0],x[1]]  # Unpack state variables
    return (np.array([v,-(2/zeta)*v - theta**n]))  # Return derivatives

def rk2(f,x,t,n,tau):                     # 2nd order Runge-Kutta method, or Midpoint method
        x_star = x + tau*f(t,x,n)/2       # Midpoint Calculation
        x = x + tau*f(t+(tau)/2,x_star,n) # Full Calculation
        return x

# ------------- #
# Solution Loop #
# ------------- #

for j in range(len(indexes)):                                       # Loop through polytropic indexes
    theta[j] = [ps(z,indexes[j]) for z in np.arange(0,ps_end+h,h)]  # Calculate the power series solution up to 1 for a given n-value
    x = np.array([theta[j][-1],dps(ps_end,indexes[j])])             # Numpy array of the initial conditions of theta and v, is required to be a NP array for vector purposes
    for i in range(len(zeta)):                                      # Loop to numerically solve for a given polytropic index
        prev_x = x                                                  # Save last solution for use in the next iteration
        x = rk2(func2,prev_x,zeta[i],indexes[j],h)                  # Solve and append
        if x[0] < 0 :                                               # Check if solution has passed through x-axis.
            break
        elif math.isnan(x[0]) == True:                              # If infinite value or non-numerical value is returned due to errors etc, remove and end solution.
            theta[j].pop()                                          
            break
        else:
            theta[j].append(x[0])                                   # Otherwise append new theta to appropriate solution array.


# ------------- #
#   Plotting    #
# ------------- #

zeta = np.append(np.arange(0,ps_end+h,h),zeta)  # Append the powerseries range and the numerical range together for plotting
analytic = [lambda z: 1 - z**2/6, lambda z: np.sin(z)/z, lambda z: (1 + z**2/3)**(-0.5)]  # Lambda functions for analytic solutions: n = 0, 1, and 5.


# Plot the numerical solutions found above

plt.plot(zeta[0:len(theta[0])],theta[0],label='n = 1.5 (Numerical)')
plt.plot(zeta[0:len(theta[1])],theta[1],label='n = 3 (Numerical)')
plt.plot(zeta[0:len(theta[2])],theta[2],label='n = 3.5 (Numerical)')


# Plot the analytic solutions using the above lamdba functions and the appropriate ranges up to the x-intercepts.

plt.plot(np.arange(0,np.sqrt(6)+h,h),analytic[0](np.arange(0,np.sqrt(6)+h,h)),label='n = 0 (Analytic)')    
plt.plot(np.arange(0,np.pi+h,h),analytic[1](np.arange(0,np.pi+h,h)),label='n = 1 (Analytic)')
plt.plot(np.arange(0,10+h,h),analytic[2](np.arange(0,10+h,h)),label='n = 5 (Analytic)')


# Other misc. plotting parameters

plt.legend(title = "Polytropic Index (Solution Type)")   # Add legend
plt.grid()  # Add grid lines.
plt.title(r'Solution of Lane-Emden Equation for Multiple Polytropic Indexes ($\theta$ vs $\zeta$)') # Add title
plt.xlabel(r'$\zeta$ (dimensionless radius)') # Add x-axis label
plt.ylabel(r'$\theta$')  # Add y-axis label
plt.show()   # Show the plot