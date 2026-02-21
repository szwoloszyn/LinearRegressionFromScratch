# LinearRegressionFromScratch

This is an implementation of chosen Linear Regression algorithms, built as a C++ library.

:exclamation: **[NOTE]**: This project was created strictly for educational purposes — to explore how linear regression works under the hood, without relying on external machine learning libraries. Therefore it is not intended for production use and I cannot guarantee that it will perform optimally or handle every possible edge case. :exclamation:

## Table of contents
- [What is it](#what-is-it)
- [How to run](#how-to-run)
- [Code Map](#code-map)
    - [Available algorithms](#1-available-algorithms)
    - [Using the library](#2-using-the-library)
    - [Library in python](#3-library-in-python)

## What is it

**LinearRegressionFromScratch** is a library written in `C++ 17`. 
It implements two basic Linear Regression algorithms - **Normal Equation** and **Batch Gradient Descent**.
Despite being written in `C++`, the library can be also used with `python >= 3.11`. 
For installation guide, visit [How to run](#howtorun) section. 
For mathematical operations, the library is using [Armadillo](https://arma.sourceforge.net/). 

This project also delivers features such as `RMSE` calculation for testing and evaluating trained model and K-fold cross validation.

## How to run
A complete installation guide, alongside with setup scripts and library usage examples, are available in [install](./install/) directory.

## Code Map

### 1. AVAILABLE ALGORITHMS:
- `Normal Equation`- Uses naive analytical [formula](https://mathworld.wolfram.com/NormalEquation.html) for linear fitting. Beware - The result may be highly inaccurate due to numerical instability, as the matrix inverted in the aforementioned formula is ill-conditioned.
- `Batch Gradient Descent` - An iterative optimization algorithm which (in this implementation) computes `RMSE function` gradient on **entire** dataset to find its `minimum` of the function.
It is very precise, but it's performance significantly decreases as the dataset grows.
- `Stochastic Gradient Descent` - ***(Not implemented yet)***. 
Works similarly to `Batch Gradient Descent` but in every iteration chooses one random sample from dataset. This way algorithm's complexity is not related to dataset size.

### 2. USING THE LIBRARY

All models are classes derived from `LinearRegression` abstract class, which provides public methods for training and evaluating models. 

To include a model in your code:
```c++
#include "modelname.h" // LOWER CASE
```
Practical example [here](./install/example.cpp).

### 2.1 `NormalEquation`

```c++
NormalEquation() 
```

`params:`

**None**

`returns:`
- **obj** : `NormalEquation`



### 2.2 `BatchGradientDescent`
- Constructor:

```c++
BatchGradientDescent::BatchGradientDescent(double eta, size_t n = 1000)
```

`parameters:`
- **eta** : `float64`
algorithm's learning rate

- **n** : `size_t`, *(optional, default = 1000)*
number of iterations

`returns:`
- **obj** : `BatchGradientDescent`

### 2.3 `LinearRegression`


```c++
void fit(const arma::mat& X, const arma::vec& y)
```
Calculates theta parameters and saves them to the selected model.

`parameters:`
- **X** : `arma::mat`
training input values (features).

- **y** : `arma::vec`
target values (labels).


`returns:`
- **None**

```c++
arma::vec predict(const arma::mat& X_pred, const arma::vec& params = arma::vec{}) const;
```
Generates predictions based on the trained model. (Optionally on provided parameters)

`parameters:`
- **X_pred** : `arma::mat`
Samples to predict.

- **params** : `arma::vec` , *(optional, default = None)*
Custom model parameters. By default uses parameters evaluated during training using `fit` method.


`returns:`
- **predictions** : `arma::vec`
Predicted labels.


```c++
std::vector<double> kFoldCrossValidation(const arma::mat& X, const arma::vec& y,
                                   const size_t k = 5) const;
```
Performs K-Fold Cross Validation on given dataset.

`parameters:`
- **X** : `arma::mat`
training input values (features).

- **y** : `arma::vec`
target values (labels).

- **k** : `size_t` , *(optional, default = 5)*
number of folds.

`returns:`
- **out** : `std::vector<double>`
`RMSE` value for each fold.


```c++
arma::vec getCoeffs() const;
```
Rturns the trained model parameters.
The model must be fitted before calling this method.

`parameters:`
- **None**

`returns:`
- **out** : `arma::vec`
Model parameters.


```c++
double getRMSE(const arma::mat& X_test, const arma::vec& y_test) const;
```
Computes **Root Mean Squared Error** for given sample and expected values.

`parameters:`
- **X_test** : `arma::mat`
test input values (features).

- **y_test** : `arma::vec`
expected values (labels).

`returns:`
- **out** : `double`
Mean error for given dataset.

```c++
void RMSEReport(const arma::mat& X_test, const arma::vec& y_test) const;
```
Prints formatted report on `RMSE` statistical parameters. *(curr only mean - TODO: add more metrics)*

`parameters:`
- **X_test** : `arma::mat`
test input values (features).

- **y_test** : `arma::vec`
expected values (labels).

`returns:`
- **None**

### 3. LIBRARY IN PYTHON

All of the above methods are available in both `C++` and `Python`. 
In order to use them in python, pass `Numpy` arrays as arguments. 
- 1D-array instead of `arma::vec` 
- ND-array instead of `arma::mat`. 
Library's `Python` package is named **linregpy**.

To use:

```python
import linregpy
```

Practical example [here](./install/example.py).

Much of the inspiration and technical understanding behind this project comes from the book [Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) written by **Aurélien Géron**.


#### FUTURE WORK
- Implement SGD
- expand RMSEReport statistics
- TODOs in code. 
