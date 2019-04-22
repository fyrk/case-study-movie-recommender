import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
sys.path.append('../')

# Function that get movie recommendations based on the cosine similarity score of movie genres
def top_sims(cosine_sim_mat, df, filter_col, movie, n):

    # Build a 1-dimensional array with movie titles
    ids = df[filter_col]
    #print(ids)
    indices = pd.Series(df.index, index=df[filter_col])
    idx = indices[movie]
    
    # Calculate Top N Similarity Scores
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices

# Function which finds the most similar movies for every movie in dataframe
def find_similarities(train_set, meta_data, n):
    # Training/DF we were given
    movie_id_col = 'id'
    genre_col = 'genres'

    movie_ids = train_set.movie.unique() #Unique ids
    
    # Make ids strings
    str_ids = map(str, movie_ids)
    str_ids = list(str_ids)

    # Secondary dataframe with more detail which is able to compute similarities
    meta_data = meta_data[meta_data[movie_id_col].isin(str_ids)]
    
    similarity = meta_data[genre_col]

    
    # Find tf_idf for items in secondary df
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tf_idf = tf.fit_transform(similarity)
    
    # Compute cosine similarity on items in secondary df
    cosine_sim_mat = linear_kernel(tf_idf, tf_idf)

    meta_data = meta_data.reset_index()
    
    # Find the top n most similar items for every movie that is in both our training dataframe and our 
    sim_list = []
    for movie in meta_data[movie_id_col].tolist():
        sim_list.append(top_sims(cosine_sim_mat, meta_data, movie_id_col, movie, n))

    meta_data['similar'] = sim_list
    meta_data[movie_id_col] = meta_data[movie_id_col].astype('int64')    
    meta_data['movie'] = meta_data[movie_id_col]
    return meta_data[['movie', 'similar']]


def drop_duplicate_movie_ratings(row, df):
    # A row-wise pandas apply function. np.NaN must be converted to -1 first
    if row['similar'] == -1:
        return row['movie']
    else:
        subset = df[df['user'] == row['user']]
        dropped_duplicates = list(set(row['similar'])-(set(row['similar'])\
                      .intersection(set(subset['movie']))))
        # add the movie from this row to the similarities matrix
        dropped_duplicates.append(row['movie'])
        return dropped_duplicates

    
def fill_ratings(training_set, n_similar_movies):
    metadata = pd.read_csv('data/the-movies-dataset/movies_metadata.csv')
    movie_similarities = find_similarities(training_set, metadata, n_similar_movies)

    #Add similar movies as column to training_set
    training_set = pd.merge(training_set, movie_similarities, on='movie', how='left')

    # drop any movies in similarities list that user has already rated
    training_set.similar = training_set.similar.fillna(-1)
    training_set.similar =  training_set.apply(drop_duplicate_movie_ratings,
                                        axis=1, args = (training_set,))

    # For each movie per user, give the 20 most similar movies the same rating
    training_set = training_set.set_index(['user', 'rating'])
    training_set = (pd.melt(training_set.similar.apply(pd.Series)
                .reset_index(), id_vars=['user', 'rating'],
                value_name='similar')
        .set_index(['user', 'rating'])
        .drop('variable', axis=1)
        .sort_index()).reset_index().dropna()
    training_set.columns = ['user', 'rating', 'movie']

    training_set.movie = training_set.movie.astype('int64')
    training_set.rating = training_set.rating.astype('float64')
    training_set.reset_index(inplace=True, drop=True)
    
    return training_set
