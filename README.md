# LinearRegressionFromScratch

This is an implementation of chosen Linear Regression algorithms, build as a C++ library.

:exclamation: NOTE: This project was created strictly for educational purposes — to explore how linear regression works under the hood, without relying on external machine learning libraries. Therefore it is not intended for production use and I cannot guarantee that it will perform optimally or handle every possible edge case. :exclamation:

## Table of contents
- [What is it](#whatisit)
- [How to run](#howtorun)
- [Code Map](#codemap)

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

All classes are extend `LinearRegression` abstract class which has implemented public methods for training and evaluating model.

### 2.1 `NormalEquation`

```
NormalEquation() 
```

`params:`

**None**

`returns:`
- **obj** : `NormalEquation`



### 2.2 `BatchGradientDescent`
- Constructor:

```
BatchGradientDescent::BatchGradientDescent(double eta, size_t n = 1000)
```
`params:`
- **eta** : `float64`
    
    algorithm's learning rate
- **n** : `size_t`

    number of iterations

`returns:`
- **obj** : `BatchGradientDescent`

### 2.2 `LinearRegression`


```
void fit(const arma::mat& X, const arma::vec& y)
```
An overriden method from base class. Calculates theta parameters and saves them to choosen object.

`params:`
- **X** : `arma::mat`

    training input values (features).
- **n** : `size_t`

    target values (labels).


`returns:`
- **None**


