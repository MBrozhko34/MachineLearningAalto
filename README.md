# MachineLearningStage2
Comparing Polynomial Regression of Different Degrees with Linear Regression for World Population Prediction

## Introduction

Population increase is an increasingly intriguing and concerning factor in modern day geopolitics. As the wealth gap between rich and poor increases, the number of people being born into developing countries as opposed to developed countries is an important factor to know. Furthermore certain countries may be experiencing decreasing populations and as such 

## Problem Formulation

In this assignment I will propose a machine learning model to predict the world population. The World's Bank dataset does not contain data points for 2021 countries' populations and as such the population of 2021 will be included in the prediction.

Datapoint: 
World population at a specific time
Features: 
Year of a country
Labels: 
Future world population at a certain year

In the case of this ML problem I will be trying to predict the future global population. The features will be the year (from 1960-2020) and the labels will be the world population at this time.

## Methods

The dataset I used is from the World Bank and as such is reliable. The page with the csv can be found at the following link: https://data.worldbank.org/indicator/SP.POP.TOTL. The dataset contains a list of populations for all countries, continents and many regions of the world as well as a summed total population of all countries and regions in the column ‘World’ (what I used). Although the dataset contains a list of populations from 1960 to 2020 for every country, I have decided to use 60 data points which is the sum of countries' population in the column ‘World’ of the dataset. Furthermore I had to transpose the dataset in order for the year to be a row instead of a column. While this may seem like a small set of data points, using more data from the dataset (i.e for each individual country) would not improve accuracy since I am just using the sum of these points instead. 

I chose the features as years from 1960 to 2020 (essentially features =  time) because it is known what the future features will be - 2020, 2021, etc and it is the quality I am measuring population against. Time increases consistently and linearly and as such is perfect for features. During the development phase I plotted time (x - feature) against population (y -  label) and it gave a clear, understandable, graphical representation of the regression problem and the prediction. In this case, time is represented as a set of integers. The population was used for labels because it is the quantity that I am trying to predict.

For this prediction, the first model I decided to use was Linear Regression. I chose this model for a few reasons. Firstly, as my task is to predict world population it is a regression problem rather than a classification task and uses numeric data. Furthermore, Linear Regression is a widely used, simplistic regression algorithm for simple problems such as population prediction. Finally, the data points show a linear relationship and progression and as such Linear Regression is appropriate.

For the second model I decided to use Polynomial Regression with degrees 3,5 and 10. I used this model because it is a good comparator to Linear Regression as it is generally more accurate as it uses polynomials instead of a linear model. Furthermore it is a more complex model which would be compared to the simple model of Linear Regression. I picked degrees 3, 5 and 10 as it is a fair comparison of different degree sizes.

For the loss functions I chose mean squared error (MSE) for both. MSE shows how close the regression line is to the set of data points. MSE is the most commonly used loss function for Linear and Polynomial Regression and as such is good for ensuring that the trained model has few outlier predictions with major errors and is also the standard loss function in the Python library (sklearn).

I split the dataset initially with 50% for training the model and then again split the remaining 50% into even halves for validation and testing so I resulted in 50% for training, 25% for validation and 25% test set. I did this because I have a dataset of 60 data points and wanted to reach a good balance of training against validating as well as making sure there were some data points left in the test set. If the validation set was too small then it would not be a representative sample for the model's performance but if it was too large the model would not have enough training or testing data.

## Results

For my results I initially compared Polynomial Regression with different degrees. 

Training Errors:
Degree 3: 182474878812554.94 
Degree 5: 186054995662141.12 
Degree 10: 195772533400225.72

Validation Errors:
Degree 3: 199037570184477.2 
Degree 5: 203196011663465.8 
Degree 10: 213729988467447.22

As shown, degree 3 had the lowest training and validation errors and as such was the best degree for Polynomial Regression. 

I then compared Polynomial Regression of degree=3 with Linear Regression and Polynomial Regression was substantially more accurate by almost a power of 10.

Linear regression:
Validation Error: 877518153423424.9
Training Error: 1655738921544125.5

Polynomial regression (Degree=3):
Validation Error: 199037570184477.2 
Training Error: 182474878812554.94 

Because of this, Polynomial Regression with Degree=3 is the chosen model for this problem as it has much lower training and validation errors than for Linear Regression.

The test set comprises 25% of the total data points from the dataset. As the chosen method is Polynomial Regression (Degree=3) this was used with the test data and yielded an error of 792354426379851.6.

## Conclusion

The results obtained from the ML model indicate that Polynomial Regression is the best model for predicting future world population. However for Linear Regression the training error is greater than validation error whereas for Polynomial Regression, although both losses are lower than for Linear Regression, the validation error is higher than the training error. This means for Polynomial Regression the model is overfitting. This could be a result of having a small dataset where a few outliers/extreme data points skew the results massively and since I am only using 15 data points (25%) for validation, this could easily occur. To try and improve this I could try to split the data in different proportions and use a much larger dataset since mine only has 60 data points.

Furthermore the test error for Polynomial Regression (Degree=3) is almost as high as the Linear Regression errors and much higher than the training and validation errors for Polynomial Regression. Again, this could be due to certain data splits with a small dataset resulting in big implications for different random data points used. To investigate and improve this I would again say using a larger dataset and testing different split proportions would be useful.


