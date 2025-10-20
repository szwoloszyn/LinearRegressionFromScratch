import linregpy
import numpy as np

np.random.seed(42)

n_samples = 1000

x1 = np.random.rand(n_samples)
x2 = np.random.rand(n_samples)

X = np.column_stack((x1, x2))

noise = np.random.randn(n_samples)
y = 3*x1 + 2*x2 + 1 + noise


a = linregpy.NormalEquation()
a.fit(X_train = X,y_train = y)
print( "ayooo:", a.predict([[67,21]], params=[0,1,2]))
print(a.getCoeffs())
print(a.kFoldCrossValidation(X=X,y=y,k=14) )

b = linregpy.BatchGradientDescent(0.1,n=20)

b.fit(y_train=y, X_train=X)
print( b.predict([[67,21]]))
print(b.getCoeffs())
print(b.kFoldCrossValidation(X=X,y=y,k=14) )
