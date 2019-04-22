import os.path
import logging
import numpy as np
import pandas as pd
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = ps.sql.SparkSession.builder \
            .master("local[4]") \
            .appName("Movie Recommender") \
            .getOrCreate()

def fill_nulls(row, df):
    # used to fill null ratings
    # A row-wise pandas apply function. np.NaN must be converted to -1 first
    if row['rating'] == -1:
        return df.loc[(df['movie'] == row['movie']), 'rating'].mean()
    else:
        return row['rating']

#best case
def fill_rating(row, movie=False):
    # null ratings must be changed to -1
    if movie==True:
        if row['rating'] == -1:
            if row['avg_mov_rating'] == -1:
                return np.NaN
            else:
                return row['avg_mov_rating']
        else:
            return row['rating']

    else:
        if row['rating'] == -1:
            if row['cluster_rating'] == -1:
                return np.NaN
            else:
                return row['cluster_rating']
        else:
            return row['rating']
    

class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.spark = (ps.sql.SparkSession.builder
                    .master("local[4]")
                    .appName("spark_Reccommender_exercise")
                    .getOrCreate()
                )
        self.sc = self.spark.sparkContext
        self.logger = logging.getLogger('reco-cs')
        # ...

    def fit(self, ratings, reg=0.27, rank=16):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")
        self.train = ratings
        df = self.spark.createDataFrame(ratings)

        self.als_ = ALS(itemCol='movie',
                userCol='user',
                ratingCol='rating',
                nonnegative=True,
                regParam=reg,
                #coldStartStrategy='drop',
                rank=rank)

        self.recommender_ = self.als_.fit(df)

        self.logger.debug("finishing fit")
        return(self)

    def transform(self, requests, cluster=True):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        #Transform pandas DF to spark DF
        request_df = self.spark.createDataFrame(requests)

        #requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        predictions = self.recommender_.transform(request_df)
        result = predictions.toPandas()
        
        #fill blanks with overall mean rating
        if cluster != True:
            predictions_df = result.fillna(self.train['rating'].mean()).rename(index=str, columns={"prediction": "rating"})
        
        # Fill blanks with predicted ratings for given cluster (best method)
        if cluster == True:
            
        # Fill blanks with average rating for that movie (second best method)
            # mov_ratings = pd.read_csv('data/training.csv')
            # avg_mov_ratings = mov_ratings.groupby('movie').mean()['rating'].reset_index()
            # avg_mov_ratings.columns = ['movie', 'avg_mov_rating']
            # result = result.rename(index=str, columns={"prediction": "rating"})
            # pred = pd.merge(result, avg_mov_ratings, on='movie', how='left')
            # pred.rating.fillna(-1, inplace=True)
            # pred['avg_mov_rating'].fillna(-1, inplace=True)
            # pred.rating = pred.apply(fill_rating, args=(True,), axis=1)
            # pred.rating.fillna(self.train.rating.mean(), inplace=True)
            # predictions_df = pred

        # Fill blanks with predicted ratings for given cluster (best method)
            cratings = pd.read_pickle('data/predicted_cluster_ratings')
            result = result.rename(index=str, columns={"prediction": "rating"})
            users = pd.read_csv('data/clustered_users.csv').drop('Unnamed: 0', axis=1)

            pred = pd.merge(result, users, on='user', how='left')
            pred2 = pd.merge(pred, cratings, how='inner', 
                        left_on=['movie', 'cluster'],
                        right_on=['movie','cluster'])
            
            pred2.rating = pred2.rating.fillna(-1)
            pred2.cluster_rating = pred2.cluster_rating.fillna(-1)

            pred2.rating = pred2.apply(fill_rating, axis=1)
            pred2.rating.fillna(self.train.rating.mean(), inplace=True)
            predictions_df = pred2

            
        #no fill for NaN
        # predictions_df = result.rename(index=str, columns={"prediction": "rating"})

        write_path = os.path.join(os.getcwd(), 'data/cpreds.csv')
        predictions_df.to_csv(write_path)
        self.logger.debug("finishing predict")
        return(predictions_df)


class RecoRegressionEvaluation(RegressionEvaluator):
    """ copy/pasted from submit.py """
    @staticmethod
    def _compute_casestudy_score(predictions, actual):
        """Look at 5% of most highly predicted movies for each user.
        Return the average actual rating of those movies.
        """
        df = pd.merge(predictions, actual, on=['user','movie']).fillna(1.0)
        #df = pd.concat([predictions.fillna(1.0), actual.actualrating], axis=1)

        # for each user
        g = df.groupby('user')

        # detect the top_5 movies as predicted by your algorithm
        top_5 = g.rating.transform(
            lambda x: x >= x.quantile(.95)
        )

        # return the mean of the actual score on those
        return df.actualrating[top_5==1].mean()

    def evaluate(self, dataset):
        # experimental
        print("evaluate based on: {}, {}".format(self.getLabelCol(),
                                                 self.getPredictionCol()))
        dataset = spark.createDataFrame(dataset)
        # create a pandas dataframe that corresponds to argument predictions
        pd_pred = dataset.select('user','movie',self.getPredictionCol())\
                         .withColumnRenamed(self.getPredictionCol(),'rating')\
                         .toPandas()
                
        # create a pandas dataframe that corresponds to argument actual
        pd_actual = dataset.select('user','movie',self.getLabelCol())\
                         .withColumnRenamed(self.getLabelCol(),'actualrating')\
                         .toPandas()
        
        # call the exact same function from submit.py
        return(self._compute_casestudy_score(pd_pred, pd_actual))


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
