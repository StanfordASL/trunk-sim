import numpy as np

def get_numpy_from_link(df, link: int):
    # get array as (6, time)
    X = None
    for dim in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        if X is None:
            X = np.array(df[f"{dim}{link}"])
        else:
            X = np.vstack((X, df[f"{dim}{link}"]))
    return X

def normalized_cross_correlation(x, y):
    #Pearson Correlation Coefficient
    x = np.array(x)
    y = np.array(y)

    # Subtract mean
    x_mean = x - np.mean(x)
    y_mean = y - np.mean(y)

    # Compute NCC
    numerator = np.sum(x_mean * y_mean)
    denominator = np.sqrt(np.sum(x_mean ** 2) * np.sum(y_mean ** 2))

    return numerator / denominator
