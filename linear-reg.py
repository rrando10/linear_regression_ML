#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RONALD RANDOLPH
# CS425: Introduction to Machine Learning
#
# Multi-Linear Regression
# Oct 9, 2018     
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#imports for data wrangling and matrices
import numpy as np
import pandas as pd

#imports for creating plots and graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#function calculates the error in current model
def cost(X, Y, B):
        l = len(Y)
        J = np.sum((X.dot(B) - Y) ** 2)/(2 * l)
        return J


#function calculates the gradient and iteratively updates the coefficients 
def gradient(X, Y, B, alpha, iterations):
        
        #initialze error history to 0
        cost_history = [0] * iterations
        l = len(Y)

        for iteration in range(iterations):
                
                #Calculate hypothesis vals
                h = X.dot(B)
                
                #Difference btw hypothesis and actual
                loss = h - Y
                
                #Calculate gradient
                gradient = (X.T.dot(loss) / l)
                
                #Update values of B w/ gradient
                B = B - alpha * gradient
                
                #New cost value
                cost = cost_function(X, Y, B)
                cost_history[iteration] = cost

        return B, cost_history


#function calculates mean square error
def rmse(Y, Y_pred):
        rmse = np.sqrt(sum((Y-Y_pred) ** 2) / len(Y))
        return rmse


#function calculates percentage error
def r2_score(Y, Y_pred):
        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - Y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def main():
        
        #opens designated file for parsing and printing of stat
        data = pd.read_csv('auto-mpg.csv')
        print(data.shape)
        data.head()

        #seperate .cvs data into arrays
        mpg = data['mpg'].values
        dsp = data['displacement'].values
        hpw = data['horsepower'].values
        wgt = data['weight'].values
        acc = data['acceleration'].values
        year = data['year'].values

        #initialize matrix of variables
        l = len(hpw)
        xO = np.ones(l)
        X = np.array([xO, dsp, year, hpw, wgt, acc]).T

        #initialize coefficients to 0
        B = np.array([0,0,0,0,0,0])

        #initialize Y (output) matrix 
        Y = np.array(mpg)
        alpha = 0.0001

        #calculate current cost (error)
        init_cost = cost(X, Y, B)
        print("Initial Cost: " + str(init_cost))

        #update coefficents for 100000 iterations
        newB, cost_history = gradient(X, Y, B, alpha, 100000)

        #print new Values of B
        print("New Coefficents: " + str(newB))
        print("Final Cost: " + str(cost_history[-1]))

        #call above functions and return results    
        Y_pred = X.dot(newB)
        print(rmse(Y, Y_pred))
        print(r2_score(Y, Y_pred))
