#-------------Documentation--------------------------------------
#The following code is the intellectual property of Alexander Polus
#This project was created for Dr. Brent Harrison's CS460G Machine Learning Class
#Start Date: 2.21.18

#In order to gain the deliverables of the project, this algorithm must be run on each synthetic dataset

#Preconditions: csv file as a command line argument to the program

#Postconditions: 4 printed error values for each 1st, 2nd, 4th, and 9th degree polynomials, along with a graph of each function generated along with its datapoints

#-------------Libraries------------------------------------------
import csv
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
#-------------Import training data-------------------------------
if (len(sys.argv) != 2):
    print("Usage: Python3 LinearRegression.py <filename> ")
    sys.exit(0)
filename = sys.argv[-1]
file = open(filename)
reader = csv.reader(file, delimiter=',')
TempData = list(reader)
TrainingData = []
for row in TempData:
    TrainingData.append([float(row[0]), float(row[1])])
print("\nTraining Data: ")
for i in TrainingData:
    print(i)

#Get my columns so I can plot the points later with MatPlotLib:
XVALUES = []
YVALUES = []
for i in range(0,len(TrainingData)):
    XVALUES.append(TrainingData[i][0])
    YVALUES.append(TrainingData[i][1])

#-------------Learning Rate--------------------------------------




Alpha = 0.0001




#-------------Define a vector multiplication function------------
def VectorMultiply(vec1, vec2):
    #throw an error if there's a logic issue and vector sizes are unequal
    result = 0
    if (len(vec1) != len(vec2)):
        print("Error, Vectors are of unequal size")
        return
    else:
        for i in range(0,len(vec1)):
            dot = vec1[i]*vec2[i]
            result += dot
    return result

#-------------Define four functions that will return my |-X-> vector given an input x---------

def FirstOrderPolynomial(x):
    values = [1, x]
    return values

def SecondOrderPolynomial(x):
    values = [1, x, x**2]
    return values

def FourthOrderPolynomial(x):
    values = [1, x, x**2, x**3, x**4]
    return values

def NinthOrderPolynomial(x):
    values = [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]
    return values

#-------------Define 4 identical functions that will take in only the training data, use global variables for theta, and will update the global variables for later use------------
###
def OutputThetaAndX1stOrder(data):
    Theta1stOrder = [0.0] * 2
    counter = 0
    while (counter <= 1000):
        for i in range(0,len(data)):
            #make deep copy of thetas for each update
            TempThetas = Theta1stOrder[:]
            CopyThetas = np.array(TempThetas,dtype=float)
            for j in range(0,len(Theta1stOrder)):
                tempX = FirstOrderPolynomial(data[i][0])
                X = np.array(tempX,dtype=float)
                Theta1stOrder[j] = CopyThetas[j] - Alpha * (np.inner(CopyThetas, X) - data[i][1]) * X[j]
        counter += 1

    return Theta1stOrder
###
def OutputThetaAndX2ndOrder(data):
    Theta2ndOrder = [0.0] * 3
    counter = 0
    while (counter <= 1000):
        for i in range(len(data)):
            #make deep copy of thetas for each update
            TempThetas = Theta2ndOrder[:]
            CopyThetas = np.array(TempThetas,dtype=float)
            for j in range(0,len(Theta2ndOrder)):
                tempX = SecondOrderPolynomial(data[i][0])
                X = np.array(tempX,dtype=float)
                Theta2ndOrder[j] = CopyThetas[j] - Alpha * (np.inner(CopyThetas, X) - data[i][1]) * X[j]
        counter += 1
    return Theta2ndOrder
###
def OutputThetaAndX4thOrder(data):
    Theta4thOrder = [0.0] * 5
    counter = 0
    while (counter <= 1000):
        for i in range(len(data)):
            #make deep copy of thetas for each update
            TempThetas = Theta4thOrder[:]
            CopyThetas = np.array(TempThetas,dtype=float)
            for j in range(0,len(Theta4thOrder)):
                tempX = FourthOrderPolynomial(data[i][0])
                X = np.array(tempX,dtype=float)
                Theta4thOrder[j] = CopyThetas[j] - Alpha * (np.inner(CopyThetas, X) - data[i][1]) * X[j]
        counter += 1
    return Theta4thOrder
###
def OutputThetaAndX9thOrder(data):
    Theta9thOrder = [0.0] * 10
    counter = 0
    while (counter <= 1000):
        for i in range(len(data)):
            #make deep copy of thetas for each update
            TempThetas = Theta9thOrder[:]
            CopyThetas = np.array(TempThetas,dtype=float)
            for j in range(0,len(Theta9thOrder)):
                tempX = NinthOrderPolynomial(data[i][0])
                X = np.array(tempX,dtype=float)
                Theta9thOrder[j] = CopyThetas[j] - (Alpha * 0.1) * (np.inner(CopyThetas, X) - data[i][1]) * X[j]
                #print("theta value ", j)
                #print(Theta9thOrder[j])
        counter += 1

    return Theta9thOrder
#-------------Error Functions------------------------------------
def FirstOrderError(input_thetas, data):
    total_error = 0
    FinalThetas = np.array(input_thetas, dtype=float)
    for row in data:
        tempX = FirstOrderPolynomial(row[0])
        X = np.array(tempX,dtype=float)
        squared_error = ( np.inner(FinalThetas, X) - row[1] ) ** 2
        total_error += squared_error
    avgError = total_error / (2 * len(data))
    return avgError
###
def SecondOrderError(input_thetas, data):
    total_error = 0
    FinalThetas = np.array(input_thetas, dtype=float)
    for row in data:
        tempX = SecondOrderPolynomial(row[0])
        X = np.array(tempX,dtype=float)
        squared_error = ( np.inner(FinalThetas, X) - row[1] ) ** 2
        total_error += squared_error
    avgError = total_error / (2 * len(data))
    return avgError
###
def FourthOrderError(input_thetas, data):
    total_error = 0
    FinalThetas = np.array(input_thetas, dtype=float)
    for row in data:
        tempX = FourthOrderPolynomial(row[0])
        X = np.array(tempX,dtype=float)
        squared_error = ( np.inner(FinalThetas, X) - row[1] ) ** 2
        total_error += squared_error
    avgError = total_error / (2 * len(data))
    return avgError
###
def NinthOrderError(input_thetas, data):
    total_error = 0
    FinalThetas = np.array(input_thetas, dtype=float)
    for row in data:
        tempX = NinthOrderPolynomial(row[0])
        X = np.array(tempX,dtype=float)
        squared_error = ( np.inner(FinalThetas, X) - row[1] ) ** 2
        total_error += squared_error
    avgError = total_error / (2 * len(data))
    return avgError
#-------------Main-----------------------------------------------

print("Running... May take up to 20 seconds")
FinalTheta1stOrder = OutputThetaAndX1stOrder(TrainingData)
FinalTheta2ndOrder = OutputThetaAndX2ndOrder(TrainingData)
FinalTheta4thOrder = OutputThetaAndX4thOrder(TrainingData)
FinalTheta9thOrder = OutputThetaAndX9thOrder(TrainingData)

print("First Order Information: ")
for i in FinalTheta1stOrder:
    print(i)
err1st = FirstOrderError(FinalTheta1stOrder, TrainingData)
print("error ", err1st)

print("\nSecond Order Information: ")
for i in FinalTheta2ndOrder:
    print(i)
err2nd = SecondOrderError(FinalTheta2ndOrder, TrainingData)
print("error ", err2nd)

print("\nFourth Order Information: ")
for i in FinalTheta4thOrder:
    print(i)
err4th = FourthOrderError(FinalTheta4thOrder, TrainingData)
print("error ", err4th)

print("\nNinth Order Information: ")
for i in FinalTheta9thOrder:
    print(i)
err9th = NinthOrderError(FinalTheta9thOrder, TrainingData)
print("error ", err9th)


#plot a graph for all four models

#plt.figure()
#plt.xlim(min(A),max(A))
#plt.ylim(min(B),max(B))
#plt.scatter(A,B)
#plt.title('Scatter Plot')
#plt.xlabel('A')
#plt.ylabel('B')
#plt.show()
















