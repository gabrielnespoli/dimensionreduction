import sys
import pandas as pd
import mylib as lib
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


def output(D):
    original = list(D.keys())[0]
    plt.clf()
    plt.ion()
    legends = []
    for d in list(D.keys())[1:]:
        usage = float(sum(D[original][0].memory_usage(index=True, deep=True))/sum(D[d][0].memory_usage(index=True, deep=True)))
        print("%.2f" % usage, "%.2f" % np.min(D[d][3]),
              "%.2f" % np.average(D[d][3]), "%.2f" % np.max(D[d][3]))
        plt.hist(D[d][3], normed=True, bins=100, histtype='step')
        legends.append('d = '+ str(d))
    plt.legend(legends)
    plt.xlabel("distortion")
    plt.ylabel("frequency")
    plt.title("Distortion after dimensionality reduction")
    plt.show(block=True)


def main():
    filename = sys.argv[1]
    dimensions = [int(x) for x in sys.argv[2:]]

    # Pivot the dataset (see pandas.DataFrame.pivot()) to employ the values of the first
    # column as index, the values of the second column as columns, and the values of the third
    # column as values (note that the columns might have any name):
    ratings = pd.read_csv(filename).pivot(index='userId', columns='movieId', values='rating')

    # Replace each NaN value with the average value of the column it belongs to
    nan_index = np.isnan(ratings)
    ratings = np.where(nan_index, np.ma.array(ratings, mask=nan_index).mean(axis=0), ratings)
    ratings = pd.DataFrame(ratings)

    # For each d in 'dimensions', use reduce() to reduce the dimensionality of the rows of the dataframe to
    # dimension d, obtaining k different reduced datasets R1, ..., Rk.
    # I used a dictionary, which the keys are the dimensions and in the values a list which will contain any data
    # related to the dimension (i.e. the original dataset, the sample points and the distance-matrix)
    D = OrderedDict()

    # add the original dataset in the dictionary with key as the original dimension
    D[len(ratings.columns)] = [ratings] # POSITION ZERO OF THE LIST
    smallest_n_points = np.inf
    for d in dimensions:
        reduced_rating = pd.DataFrame(data=lib.reduce(ratings, d))
        D[d] = [reduced_rating]

        # get the number of points of the smallest dataset
        smallest_n_points = min(250, reduced_rating.shape[0], smallest_n_points)

    # Then, from the original dataset and from each reduced dataset R1, ..., Rk, sample a
    # subset of min{250, n} random points, where n is the number of points in each dataset.
    random_rows = np.random.choice(smallest_n_points, size=min(250, smallest_n_points), replace=False)

    # create a matrix X, which will contain the subset of sample points
    for d in D.keys():
        df = D[d][0]
        sample = df.iloc[random_rows]
        # put in the list of the dimension the sample points - POSITION ONE OF THE LIST
        D[d].append(sample)

        # get the all-to-all distances - POSITION TWO OF THE LIST
        D[d].append(lib.alldist(sample))

    # get the distortions of each reduced matrix and the original matrix
    original_dimension = list(D.keys())[0]
    for d in list(D.keys())[1:]:
        # calculate the distortion between the distance matrix of each dimension and the original one
        D[d].append(lib.distortion(D[original_dimension][2], D[d][2]))   # POSITION THREE OF THE LIST

    output(D)


if __name__ == "__main__":
    main()
