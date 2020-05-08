import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':

    # (1)
    # read data
    df = pd.read_csv('data/data2.csv')

    # create X, y
    x = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    # fit model
    model = LinearRegression()
    model.fit(x, y)

    plt.figure(figsize=(15, 15))
    idx = 1
    for degree in [2, 3, 4, 5, 8, 10, 12, 15]:
        start = time.time()

        plt.subplot(3, 3, idx)

        poly = PolynomialFeatures(degree=degree)
        x_p = poly.fit_transform(x)
        poly.fit(x_p, y)
        poly_lin = LinearRegression()
        poly_lin.fit(x_p, y)

        duration = time.time() - start
        print("Duration of training for degrees={}: {} secs.".format(degree, duration))

        plt.scatter(x, y, color='blue')
        plt.plot(x, model.predict(x), color='red')
        plt.plot(x, poly_lin.predict(poly.fit_transform(x)), color='green')
        plt.title('degree = {}'.format(degree))
        idx += 1
    plt.savefig("poly_degrees.png")
    plt.show()

    # (2)
    # read data
    df = pd.read_csv('data/data2b.csv')

    # create X, y
    x = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    # fit model
    model = LinearRegression()
    model.fit(x, y)

    degree = 15
    start = time.time()

    poly = PolynomialFeatures(degree=degree)
    x_p = poly.fit_transform(x)
    poly.fit(x_p, y)
    poly_lin = LinearRegression()
    poly_lin.fit(x_p, y)

    duration = time.time() - start
    print("Duration of training for degrees={}: {} secs.".format(degree, duration))

    plt.scatter(x, y, color='blue')
    plt.plot(x, model.predict(x), color='red')
    plt.plot(x, poly_lin.predict(poly.fit_transform(x)), color='green')
    plt.savefig("fig2_d_{}.png".format(degree))
    plt.show()
