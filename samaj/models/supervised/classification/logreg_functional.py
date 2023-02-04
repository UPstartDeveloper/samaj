# %% [markdown]
# # Programming Assignment: Numerical Optimization for Logistic Regression.
# This was completed for CS 583 "Deep Learning" at Stevens.
# The instructor who provided much of the starter code for this work is Justo Karell: https://justokarell.com/

# TODO[Zain]: make an object-oriented version of this model
    # it would ideally reuse some of the logic/code in the
    # "LinearRegressorGD" class
# 

# %% [markdown]
# ## 0. You will do the following:
# 
# 1. Read the lecture note: [click here](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Logistic/paper/logistic.pdf)
# 
# 2. Read, complete, and run my code.
# 
# 3. **Implement mini-batch SGD** and evaluate the performance.
# 
# 4. Convert the .IPYNB file to .HTML file.
# 
#     * The HTML file must contain **the code** and **the output after execution**.
#     
#     * Missing **the output after execution** will not be graded.
#     
#     
# 5. Upload this .HTML file to your Google Drive, Dropbox, or your Github repo.  (If you submit the file to Google Drive or Dropbox, you must make the file "open-access". The delay caused by "deny of access" may result in late penalty.)
# 
# 6. On Canvas, submit the Google Drive/Dropbox/Github link to the HTML file.
# 
# 
# ## Grading criteria:
# 
# 1. When computing the ```gradient``` and ```objective function value``` using a batch of samples, use **matrix-vector multiplication** rather than a FOR LOOP of **vector-vector multiplications**.
# 
# 2. Plot ```objective function value``` against ```epochs```. In the plot, compare GD, SGD, and MB-SGD (with $b=8$ and $b=64$). The plot must look reasonable.

# %% [markdown]
# # 1. Data processing
# 
# - Download the Diabete dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes
# - Load the data using sklearn.
# - Preprocess the data.

# %% [markdown]
# ## 1.1. Load the data

# %%
from sklearn import datasets
import numpy as np

x_sparse, y = datasets.load_svmlight_file('diabetes.txt')  # note: changed from 'diabetes'
x = x_sparse.todense()

print('Shape of x: ' + str(x.shape))  # 8 features
print('Shape of y: ' + str(y.shape))

# %% [markdown]
# ## 1.2. Partition to training and test sets
# 
# Here we are using about 83%, or 5/6 of the total no. of samples, for training.

# %%
np.random.seed(42)  # for reproducibility purposes

# %%
# partition the data to training and test sets
n = x.shape[0]
n_train = 640
n_test = n - n_train

rand_indices = np.random.permutation(n)
train_indices = rand_indices[0:n_train]
test_indices = rand_indices[n_train:n]

x_train = x[train_indices, :]
x_test = x[test_indices, :]
y_train = y[train_indices].reshape(n_train, 1)
y_test = y[test_indices].reshape(n_test, 1)

print('Shape of x_train: ' + str(x_train.shape))
print('Shape of x_test: ' + str(x_test.shape))
print('Shape of y_train: ' + str(y_train.shape))
print('Shape of y_test: ' + str(y_test.shape))

# %% [markdown]
# ## 1.3. Feature scaling

# %% [markdown]
# Use standardization to transform both training and test features:

# %%
# calculate mu and sig using the training set
d = x_train.shape[1]
mu = np.mean(x_train, axis=0).reshape(1, d)
sig = np.std(x_train, axis=0).reshape(1, d)

# transform the training features
x_train = (x_train - mu) / (sig + 1E-6)  # what is this "1E-6" term doing here?

# transform the test features - using what we learned from the training set only
x_test = (x_test - mu) / (sig + 1E-6)

print('test mean = ')
print(np.mean(x_test, axis=0))

print('test std = ')
print(np.std(x_test, axis=0))

# %% [markdown]
# ## 1.4. Add a dimension of all ones

# %%
n_train, d = x_train.shape
x_train = np.concatenate((x_train, np.ones((n_train, 1))), axis=1)  # adds a col of 1's

n_test, d = x_test.shape
x_test = np.concatenate((x_test, np.ones((n_test, 1))), axis=1)

print('Shape of x_train: ' + str(x_train.shape))
print('Shape of x_test: ' + str(x_test.shape))

# %% [markdown]
# # 2. Logistic regression model
# 
# The objective function is $Q (w; X, y) = \frac{1}{n} \sum_{i=1}^n \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.

# %%
# Calculate the objective function value
# Inputs:
#     w: d-by-1 matrix
#     x: n-by-d matrix
#     y: n-by-1 matrix
#     lam: scalar, the regularization parameter
# Return:
#     objective function value (scalar)
def objective(w, x, y, lam):
    n, d = x.shape
    yx = np.multiply(y, x) # n-by-d matrix
    yxw = np.dot(yx, w) # n-by-1 matrix
    vec1 = np.exp(-yxw) # n-by-1 matrix
    vec2 = np.log(1 + vec1) # n-by-1 matrix
    loss = np.mean(vec2) # scalar
    reg = lam / 2 * np.sum(w * w) # scalar
    return loss + reg
    

# %%
# initialize w
d = x_train.shape[1]
w = np.zeros((d, 1))

# evaluate the objective function value at w
lam = 1E-6
objval0 = objective(w, x_train, y_train, lam)
print('Initial objective function value = ' + str(objval0))  # aka the training error (regularized)

# %% [markdown]
# # 3. Numerical optimization

# %% [markdown]
# ## 3.1. Gradient descent
# 

# %% [markdown]
# The gradient at $w$ is $g = - \frac{1}{n} \sum_{i=1}^n \frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$

# %%
# Calculate the gradient
# Inputs:
#     w: d-by-1 matrix
#     x: n-by-d matrix
#     y: n-by-1 matrix
#     lam: scalar, the regularization parameter
# Return:
#     g: g: d-by-1 matrix, full gradient
def gradient(w, x, y, lam):
    n, d = x.shape
    yx = np.multiply(y, x) # n-by-d matrix
    yxw = np.dot(yx, w) # n-by-1 matrix
    vec1 = np.exp(yxw) # n-by-1 matrix
    vec2 = np.divide(yx, 1+vec1) # n-by-d matrix
    vec3 = -np.mean(vec2, axis=0).reshape(d, 1) # d-by-1 matrix
    g = vec3 + lam * w
    return g

# %%
# Gradient descent for solving logistic regression
# Inputs:
#     x: n-by-d matrix
#     y: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     stepsize: scalar
#     max_iter: integer, the maximal iterations
#     w: d-by-1 matrix, initialization of w
# Return:
#     w: d-by-1 matrix, the solution
#     objvals: a record of each iteration's objective value
def grad_descent(x, y, lam, stepsize, max_iter=100, w=None):
    n, d = x.shape
    objvals = np.zeros(max_iter) # store the objective values
    if w is None:
        w = np.zeros((d, 1)) # zero initialization of the weights
    
    for t in range(max_iter):
        objval = objective(w, x, y, lam)
        objvals[t] = objval
        print('Objective value at t=' + str(t) + ' is ' + str(objval))
        g = gradient(w, x, y, lam)
        w -= stepsize * g
    
    return w, objvals

# %% [markdown]
# Run gradient descent.

# %%
lam = 1E-6
stepsize = 1.0
w, objvals_gd = grad_descent(x_train, y_train, lam, stepsize)

# %% [markdown]
# ## 3.2. Stochastic gradient descent (SGD)
# 
# Define $Q_i (w) = \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.
# 
# The stochastic gradient at $w$ is $g_i = \frac{\partial Q_i }{ \partial w} = -\frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

# %%
# Calculate the objective Q_i and the gradient of Q_i
# Inputs:
#     w: d-by-1 matrix
#     xi: 1-by-d matrix
#     yi: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i
def stochastic_objective_gradient(w, xi, yi, lam):
    yx = yi * xi # 1-by-d matrix
    yxw = float(np.dot(yx, w)) # scalar
    
    # calculate objective function Q_i
    loss = np.log(1 + np.exp(-yxw)) # scalar
    reg = lam / 2 * np.sum(w * w) # scalar
    obj = loss + reg
    
    # calculate stochastic gradient
    g_loss = -yx.T / (1 + np.exp(yxw)) # d-by-1 matrix
    g = g_loss + lam * w # d-by-1 matrix
    
    return obj, g

# %%
# SGD for solving logistic regression
# Inputs:
#     x: n-by-d matrix
#     y: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     stepsize: scalar
#     max_epoch: integer, the maximal epochs
#     w: d-by-1 matrix, initialization of w
# Return:
#     w: the solution
#     objvals: record of each iteration's objective value
def sgd(x, y, lam, stepsize, max_epoch=100, w=None):
    n, d = x.shape
    objvals = np.zeros(max_epoch) # store the objective values
    if w is None:
        w = np.zeros((d, 1)) # zero initialization
    
    for t in range(max_epoch):
        # randomly shuffle the samples
        rand_indices = np.random.permutation(n)
        x_rand = x[rand_indices, :]
        y_rand = y[rand_indices, :]
        
        objval = 0 # accumulate the objective values
        for i in range(n):
            xi = x_rand[i, :] # 1-by-d matrix - b/c SGD only needs 1 sample --> gradient
            yi = float(y_rand[i, :]) # scalar
            obj, g = stochastic_objective_gradient(w, xi, yi, lam)
            objval += obj
            w -= stepsize * g
        
        stepsize *= 0.9 # decrease step size
        objval /= n
        objvals[t] = objval
        print('Objective value at epoch t=' + str(t) + ' is ' + str(objval))
    
    return w, objvals

# %% [markdown]
# Run SGD.

# %%
lam = 1E-6
stepsize = 0.1
w, objvals_sgd = sgd(x_train, y_train, lam, stepsize)

# %% [markdown]
# # 4. Compare GD with SGD
# 
# Plot objective function values against epochs.

# %%
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(6, 4))

epochs_gd = range(len(objvals_gd))
epochs_sgd = range(len(objvals_sgd))

line0, = plt.plot(epochs_gd, objvals_gd, '--b', linewidth=4)
line1, = plt.plot(epochs_sgd, objvals_sgd, '-r', linewidth=2)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Objective Value', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend([line0, line1], ['GD', 'SGD'], fontsize=20)
plt.tight_layout()
plt.show()
fig.savefig('compare_gd_sgd.pdf', format='pdf', dpi=1200)

# %% [markdown]
# # 5. Prediction

# %%
# Predict class label
# Inputs:
#     w: d-by-1 matrix
#     X: m-by-d matrix
# Return:
#     f: m-by-1 matrix, the predictions
def predict(w, X):
    xw = np.dot(X, w)
    f = np.sign(xw)
    return f

# %%
# evaluate training error
f_train = predict(w, x_train)
diff = np.abs(f_train - y_train) / 2
error_train = np.mean(diff)
print('Training classification error is ' + str(error_train))

# %%
# evaluate test error
f_test = predict(w, x_test)
diff = np.abs(f_test - y_test) / 2
error_test = np.mean(diff)
print('Test classification error is ' + str(error_test))

# %% [markdown]
# # 6. Mini-batch SGD (fill the code)
# 
# 

# %% [markdown]
# ## 6.1. Compute the objective $Q_I$ and its gradient using a batch of samples
# 
# Define $Q_I (w) = \frac{1}{b} \sum_{i \in I} \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $, where $I$ is a set containing $b$ indices randomly drawn from $\{ 1, \cdots , n \}$ without replacement.
# 
# The stochastic gradient at $w$ is $g_I = \frac{\partial Q_I }{ \partial w} = \frac{1}{b} \sum_{i \in I} \frac{- y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

# %%
# Calculate the objective Q_I and the gradient of Q_I
# Inputs:
#     w: d-by-1 matrix
#     xi: b-by-d matrix
#     yi: b-by-1 matrix
#     lam: scalar, the regularization parameter
#     b: integer, the batch size
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i
def mb_stochastic_objective_gradient(w, xi, yi, lam, b):
    yx = np.multiply(yi, xi)  # b-by-d matrix
    yxw = np.dot(yx, w).astype(np.float64)  # b-by-1
    
    # calculate objective function Q_i
    loss = np.log(1 + np.exp(-yxw)) # b-by-1
    reg = lam / 2 * np.sum(w * w) # b-by-1
    obj = (loss + reg).mean()
    
    # calculate stochastic gradient
    g_loss = (-yx / (1 + np.exp(yxw))).T # d-by-b matrix
    g = (g_loss + lam * w).mean(axis=1) # d-by-1 matrix
    # obj, g = stochastic_objective_gradient(w, xi, yi, lam)
    
    return obj, g
    

# %% [markdown]
# ## 6.2. Implement mini-batch SGD
# 
# Hints:
# 1. In every epoch, randomly permute the $n$ samples (just like SGD).
# 2. Each epoch has $\frac{n}{b}$ iterations. In every iteration, use $b$ samples, and compute the gradient and objective using the ``mb_stochastic_objective_gradient`` function. In the next iteration, use the next $b$ samples, and so on.
# 

# %%
# Mini-Batch SGD for solving logistic regression
# Inputs:
#     x: n-by-d matrix
#     y: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     b: integer, the batch size
#     stepsize: scalar
#     max_epoch: integer, the maximal epochs
#     w: d-by-1 matrix, initialization of w
# Return:
#     w: the solution
#     objvals: record of each iteration's objective value
def mb_sgd(x, y, lam, b, stepsize, max_epoch=100, w=None):
    # Fill the function
    # Follow the implementation of sgd
    # Record one objective value per epoch (not per iteration!)
    n, d = x.shape
    objvals = np.zeros(max_epoch) # store the objective values
    if w is None:
        w = np.zeros((d, 1)) # zero initialization
    
    for t in range(max_epoch):
        # randomly shuffle the samples
        rand_indices = np.random.permutation(n)
        x_rand = x[rand_indices, :]
        y_rand = y[rand_indices, :]
        
        objval = 0 # accumulate the objective values
        for i in range(0, n, b):
            xi = x_rand[i:i+b, :] # b-by-d matrix
            yi = y_rand[i:i+b, :].astype(np.float64)  # b-by-1 vector
            obj, g = mb_stochastic_objective_gradient(w, xi, yi, lam, b)
            objval += obj
            w -= stepsize * g
        
        stepsize *= 0.9 # decrease step size
        objval /= (n/b)
        objvals[t] = objval
        print('Objective value at epoch t=' + str(t) + ' is ' + str(objval))
    
    return w, objvals

# %% [markdown]
# ## 6.3. Run MB-SGD

# %%
# MB-SGD with batch size b=8
lam = 1E-6 # do not change
b = 8 # do not change
stepsize = 0.7 # was initially at 0.1

w, objvals_mbsgd8 = mb_sgd(x_train, y_train, lam, b, stepsize)

# %%
# MB-SGD with batch size b=64
lam = 1E-6 # do not change
b = 64 # do not change
stepsize = 0.5 # was initially at 0.1

w, objvals_mbsgd64 = mb_sgd(x_train, y_train, lam, b, stepsize)

# %% [markdown]
# # 7. Plot and compare GD, SGD, and MB-SGD

# %% [markdown]
# You are required to compare the following algorithms:
# 
# - Gradient descent (GD)
# 
# - SGD
# 
# - MB-SGD with b=8
# 
# - MB-SGD with b=64
# 
# Follow the code in Section 4 to plot ```objective function value``` against ```epochs```. There should be four curves in the plot; each curve corresponds to one algorithm.

# %% [markdown]
# Hint: Logistic regression with $\ell_2$-norm regularization is a strongly convex optimization problem. All the algorithms will converge to the same solution. **In the end, the ``objective function value`` of the 4 algorithms will be the same. If not the same, your implementation must be wrong. Do NOT submit wrong code and wrong result!**

# %%
# plot the 4 curves:
fig = plt.figure(figsize=(12, 8))

epochs_gd = range(len(objvals_gd))
epochs_sgd = range(len(objvals_sgd))
epochs_mbsgd8 = range(len(objvals_mbsgd8))
epochs_mbsgd64 = range(len(objvals_mbsgd64))

line0, = plt.plot(epochs_gd, objvals_gd, '--b', linewidth=4)
line1, = plt.plot(epochs_sgd, objvals_sgd, '-r', linewidth=2)
line2, = plt.plot(epochs_mbsgd8, objvals_mbsgd8, '+g', linewidth=5)
line3, = plt.plot(epochs_mbsgd64, objvals_mbsgd64, '-y', linewidth=3)

plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Objective Value', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(
    [line0, line1, line2, line3],
    ['GD', 'SGD', 'MBSGD8', 'MBSGD64'], 
    fontsize=20
)
plt.tight_layout()
plt.show()
fig.savefig('compare_gd_sgd_mbsgd.pdf', format='pdf', dpi=1200)


