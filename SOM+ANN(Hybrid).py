#Combining SOM and ANN to find probabilities of customers being frauds and 
#also ranking of those probabilities

#Part 1 - Identify the frauds with SOM

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
data = pd.read_csv("Credit_Card_Applications.csv") 
x = data.iloc[:,:-1].values 
y = data.iloc[:,-1].values 

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)
#Training the SOM - implement SOM from scratch or use a class already developed
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5) 
som.random_weights_init((x))
som.train_random(data=x,num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,v in enumerate(x): 
    w = som.winner(v) 
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show() 
mappings = som.win_map(x) 
fraud = np.concatenate((mappings[(3,1)],mappings[(7,1)],mappings[(6,2)],
                    mappings[(8,2)],mappings[(6,4)],mappings[(2,8)],mappings[(3,8)]),axis=0)
fraud = sc.inverse_transform(fraud)

#Part 2 - Going from Unsupervised to Supervised DL

#Creating the matrix of features
customers = data.iloc[:,1:].values
#CustomerId will not help in predicting probabilities
#Class is relevant info which mcan help in finding correlations between 
#customer info and its probability to cheat

#Creating the dependent variable(this is a supervised model)
#Dependent variable consists 0->not fraud & 1->fraud
#we can create it using 'fraud variable which contains frauds
#So,those customers will be assigned 1 and others 0
is_fraud = np.zeros(len(data))#dependent variable vector with zeroes of data(690) long
#count=0
for i in range(len(data)):#default start is 0 so,only stop is specified
    #This loop will compare CustomerIds(data,fraud) and replace with 1's when matched
    if data.iloc[i,0] in fraud:# compares CustomerIds
        is_fraud[i] = 1
#        count=count+1
#print(count)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)# no need to scale is_fraud as it is already in [0,1]

# Initializing the ANN
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()

#Making the ANN model simple as 690 is very small for DL 
# Adding the input layer and the first hidden layer
ann.add(Dense(units=2, kernel_initializer='uniform', activation='relu', 
                              input_dim=15))

# Adding the output layer
ann.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(customers, is_fraud, batch_size = 1, epochs = 2)

#Predicting the probabilities
y_pred = ann.predict(customers)
#2 options to rank the probabilities
#1) export y_pred to excel and rank the probabilities
#2) using python

y_pred = np.concatenate((data.iloc[:,0:1].values,y_pred),axis=1)#concatenating CustomerId and y_pred
#to concat,both the elements must be of same dimension like data->2D and y_pred->2D
#data is made 2D array by range 0:1 and .values

#Now to sort, there is sort() but it will sort both the columns.So,we are using argsort()
y_pred = y_pred[y_pred[:,1].argsort()]#[:,1]all rows of 2 column are sorted
