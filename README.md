# Samaj

![Samaj logo](https://i.postimg.cc/XNchcYWp/86fd3329e5dd42189d53b2960eb80c7e-3.png)

Implementing classic machine learning algorithms from scratch with NumPy.

Fun fact: "samaj" is a transliteration of the word for "understanding" in Urdu/Hindi!

## How to Run Locally

For best results, use Python 3.9.
Please use `python -m pip install -r requirements.txt` to install the dependencies.

## What's Included Here

1. **Exploratory Data Analysis** - see the `analysis` package.
1. **Linear/Polynomial Regression via OLS** - see the `models.supervised.regression.ols` and `models.supervised.regression.linear_via_backprop` modules.
1. **Unsupervised Learning via K-Means Clustering** - see the `models.unsupervised.clustering.kmeans` module.
1. For supervised models that use backprop: we have `monitor.py`, to **record the learning curves**.
1. **Optimizers**: so far has `gradient_descent`.
