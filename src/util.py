import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.recommender import MovieRecommender

def random_subset(sample_size):
    fname = 'data/training.csv'
    n = sum(1 for line in open(fname)) - 1
    s = sample_size
    skip = sorted(random.sample(range(1, n+1), n-s))
    subset = pd.read_csv(fname, skiprows=skip, usecols=['user', 'movie', 'rating'])
    return subset

def generate_file_split(df, tt_ratio):
    #df.drop('timestamp', axis=1, inplace=True)
    s = len(df.index)
    split_index = int(tt_ratio * s)
    train = df.iloc[range(split_index)]
    test = df.iloc[range(split_index, s)].rename(index=str, columns={"rating": "actualrating"})
    requests = test.drop(columns='actualrating')
    train.to_csv('data/ctrain.csv', index=False)
    test.to_csv('data/ctest.csv', index=False)
    requests.to_csv('data/crequests.csv', index=False)

def RMSE(pd_predictions, pd_test):
    yhat = pd_predictions.rating.values
    y = pd_test.actualrating.values
    
    rmse = np.sqrt(np.sum(np.power((yhat-y),2))/len(pd_predictions))
    return rmse

def violin_plot(pd_predictions, pd_test):
    data = [pd_predictions['rating'][pd_test['actualrating'] == rating].values for rating in range(1, 6)]

    plt.violinplot(data, range(1,6), showmeans=True)
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('True vs. ALS Recommender Predicted Ratings')
    plt.show()


def grid_search(train_data, test_data, request_data, maxIter, regParams, ranks):
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # train ALS model
            reco_instance = MovieRecommender()
            fit_model = reco_instance.fit(train_data, reg, rank=rank)
            # evaluate the model by computing the RMSE on the validation data
            predictions = reco_instance.transform(request_data)
            rmse = RMSE(predictions, test_data)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = fit_model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model




