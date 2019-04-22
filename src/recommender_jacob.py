import os.path
import logging
import numpy as np
import pandas as pd
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS


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

    def fit(self, ratings):
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
                regParam=0.1,
                #coldStartStrategy='drop',
                rank=10)

        self.recommender_ = self.als_.fit(df)

        self.logger.debug("finishing fit")
        return(self)

    def transform(self, requests, similarity_df):
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
        return result
        result = pd.merge(result, similarity_df, how='outer', on='movie')
        na_df = result[result['prediction'].isna()]
        na_df = na_df.reset_index()
        filled_df = result[result['prediction'].notnull()]
        filled_df = filled_df.reset_index()
        for idx, user in enumerate(na_df['user'].tolist()):
            avg_list = []
            #print(na_df.loc[idx]['similar'])
            for m in na_df.loc[idx]['similar']:
                user_rating = self.train[(self.train['user'] == user) & (self.train['movie'] == m)]
                try:
                    user_rating = user_rating.reset_index()
                    avg_list.append(user_rating.loc[0]['prediction'])
                except:
                    pass
            if len(avg_list) == 0:
                na_df.loc[idx]['prediction'] = self.train['rating'].mean()
            else:    
                na_df.loc[idx]['prediction'] = sum(avg_list) / len(avg_list)
        
        predictions_df = pd.concat([filled_df, na_df], axis=0)
        predictions_df = predictions_df.reset_index()
        write_path = os.path.join(os.getcwd(), 'data/cpreds.csv')
        predictions_df.to_csv(write_path)
        self.logger.debug("finishing predict")
        return(predictions_df)


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
