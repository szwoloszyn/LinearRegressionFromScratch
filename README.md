# LinearRegressionFromScratch

This is an implementation of chosen Linear Regression algorithms, built as a C++ library.

:exclamation: NOTE: This project was created strictly for educational purposes — to explore how linear regression works under the hood, without relying on external machine learning libraries. Therefore it is not intended for production use and I cannot guarantee that it will perform optimally or handle every possible edge case. :exclamation:

## Table of contents
- [What is it](#what-is-it)
- [How to run](#how-to-run)
- [Code Map](#code-map)
    - [Available algorithms](#1-available-algorithms)
    - [Using the library](#2-using-the-library)
    - [Library in python](#3-library-in-python)

## What is it

**LinearRegressionFromScratch** is a library written in `C++ 17`. It implements two basic Linear Regression algorithms - *Normal Equation and Batch Gradient Descent.* Library might be built int. Despite the fact that the library is written in `C++` language, it might be also used with `python >= 3.11`. For installation guide, visit [How to run](#howtorun) section. For mathematical object representation, library is using [Armadillo](https://arma.sourceforge.net/). This project also delivers features such as RMSE calculation for testing and evaluating trained model and K-fold cross validation.

## How to run
Full installation guide, alongside with installation scripts and library usage examples, are available in [install](./install/) directory.

## Code Map

### 1. AVAILABLE ALGORITHMS:
- `Normal Equation`- defined [formula](https://mathworld.wolfram.com/NormalEquation.html) for linear fit. Beware - its accurate but **very** slow when number of features goes up. 
- `Batch Gradient Descent` - iterative algorithm which (in this implementation) calculates `RMSE function` gradient on **whole** dataset in order to find `minimum` of the function. It is very precise, but it's performance is significantly dropping, when the dataset is getting large.
- `Stochastic Gradient Descent` - **(Not implemented yet)**. Works like `Batch Gradient Descent` but in every iteration chooses one random entry from dataset. This way algorithm's complexity is not related to size of a dataset.

### 2. USING THE LIBRARY

All classes are derived from `LinearRegression` abstract class which has implemented public methods for training and evaluating model. These can be used with any model.

In order to use a model in your code, use:
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

`params:`
- **eta** : `float64`
algorithm's learning rate

- **n** : `size_t`
number of iterations

`returns:`
- **obj** : `BatchGradientDescent`

### 2.3 `LinearRegression`


```c++
void fit(const arma::mat& X, const arma::vec& y)
```
Calculates theta parameters and saves them to choosen model.

`params:`
- **X** : `arma::mat`
training input values (features).

- **y** : `arma::vec`
target values (labels).


`returns:`
- **None**

```c++
arma::vec predict(const arma::mat& X_pred, const arma::vec& params = arma::vec{}) const;
```
Model predictions based on `RMSE` function. 

`params:`
- **X_pred** : `arma::mat`
Samples to be predicted.

- **params** : `arma::vec` , **optional**
Manual model parameters. By default uses parameters evaluated during training using `fit` method.


`returns:`
- **predictions** : `arma::vec`
Predicted labels.


```c++
std::vector<double> kFoldCrossValidation(const arma::mat& X, const arma::vec& y,
                                   const size_t k = 5) const;
```
Performs K-Fold cross validation on given dataset.

`params:`
- **X** : `arma::mat`
training input values (features).

- **y** : `arma::vec`
target values (labels).

- **k** : `size_t` , **optional**
number of folds. Default value is `k = 5`

`returns:`
- **out** : `std::vector<double>`
`RMSE` for each fold.


```c++
arma::vec getCoeffs() const;
```
Returns model parameters. Model needs to be trained in order to call this method

`params:`
- **None**

`returns:`
- **out** : `arma::vec`
Model parameters.


```c++
double getRMSE(const arma::mat& X_test, const arma::vec& y_test) const;
```
Returns RMSE error for given sample.

`params:`
- **X** : `arma::mat`
testing input values (features).

- **y** : `arma::vec`
expected values (labels).

`returns:`
- **out** : `double`
Mean error for given dataset.

```c++
void RMSEReport(const arma::mat& X_test, const arma::vec& y_test) const;
```
Prints clean report on `RMSE` statistical parameters. (just mean for now TODO later :) )

`params:`
- **X** : `arma::mat`
testing input values (features).

- **y** : `arma::vec`
expected values (labels).

`returns:`
- **None**

### 3. LIBRARY IN PYTHON

All of the above methods are available in bot `C++` and `Python`. In order to use them in python, pass `Numpy` arrays as arguments. 1D-array instead of `arma::vec` and ND-array instead of `arma::mat`. Library's `Python` package is named **linregpy**.

Simply import the package in python file and all the classes and methods are available.

```python
import linregpy
```

Practical example [here](./install/example.py).
