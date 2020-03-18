import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, euclidean

# function that creates a list containing check-in points for a individual user
def getPoint(userid, csv_path):
    df = pd.read_csv(csv_path)
    newdf = df[df["user_id"] == userid]
    newdf['combined'] = list(zip(newdf.Latitude, newdf.Longitude))
    points = newdf['combined'].values.tolist()
    return points

# python method for geometric median
def method1(points):
    distance = euclidean
    geometric_mediod = min(map(lambda p1:(p1,sum(map(lambda p2:distance(p1,p2),points))),points), key = lambda x:x[1])[0]
    return geometric_mediod


# weiszfeld algorithm
def weiszfeld_method(points):
    # change to array
    points = np.asarray(points)

    # set tolerance
    options = {'maxiter': 1000, 'tol': 1e-7}

    # calculate euclidean distance
    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)
    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # avoid to be divided by zero
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points / distances).sum(axis=0) / (1. / distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next) ** 2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


# use minimize method in scipy.optimize
def minimize_method(points):
    # change to array
    points = np.asarray(points)

    # calculate euclidean distance
    def sum_distance_func(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    guess = points.mean(axis=0)

    optimize_result = minimize(sum_distance_func, guess, method='COBYLA')
    return optimize_result.x