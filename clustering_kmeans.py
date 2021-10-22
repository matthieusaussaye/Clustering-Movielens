import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import helper
from sklearn.cluster import KMeans 


class ClusteringMovies() :
    """
    Thiss class handle all the clustering process for the flask application :
    - Preprocessing
    - Filtering
    - Modelling
    - User Embedding analytics
    
    """

    def __init__(self) :

        self.data = pd.read_csv('./data/u.data', sep="\t", header=None)
        self.item = pd.read_csv('./data/u.item', sep="|", encoding='latin-1', header=None)
        self.predictions = None
        self.most_rated_movies_1k = pd.DataFrame()
        self.sparse_ratings = pd.DataFrame()
    
    def pre_processing(self) : 
        """ Extract rating features from the database """

        self.data.columns = ['userId', 'movie id', 'rating', 'timestamp'] 
        self.item.columns = ['movie id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
            'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        user_ratings = pd.merge(self.data[['userId','movie id','rating']], self.item[['movie id','title']], on="movie id")
        user_ratings = user_ratings.drop('movie id',axis=1)
        user_ratings = user_ratings.pivot_table(values = 'rating', columns = "title", index = 'userId')

        return(user_ratings)
    
    def filtering(self,user_ratings) :
        """ Filtering features by relevance (1000 most rated movies) """

        self.most_rated_movies_1k = helper.get_most_rated_movies(user_ratings, 1000)
        dtype=pd.SparseDtype(float)
        self.sparse_ratings = csr_matrix(self.most_rated_movies_1k.astype(dtype).sparse.to_coo())

        return(self.sparse_ratings)
    
    def kmeans_modelling(self) :
        """ Modelisation using the kmeans. """
        # 20 clusters
        user_ratings = self.pre_processing()
        sparse_ratings = self.filtering(user_ratings)
        self.predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

        return self.predictions

    def user_embedding(self,
                       user_id : int,
                       max_users : int,
                       max_movies : int) : 

        """ Return the cluster index & heatmap (ratings) of a user embedding """

        clustered = pd.concat([self.most_rated_movies_1k.reset_index(), pd.DataFrame({'group':self.predictions})], axis=1)


        return helper.draw_user_embedding(clustered=clustered,
                                          user_id=user_id,
                                          max_users=max_users,
                                          max_movies=max_movies)




