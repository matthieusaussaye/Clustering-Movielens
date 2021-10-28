import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score
import base64
from io import BytesIO

        
def clustering_errors(k : int,
                      data) -> list :
    """
    Return the silhouette score of k-means
    
    Return:
        - silhouette_avg (list) : list of silhouette scores.
    """
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg
    
def draw_movie_clusters(clustered : list,
                        max_users : int,
                        max_movies : int):
    """
    Draw every interesting clusters heatmap (max_users, max_movies)

    Args:
        - clustered (list) : list of the clusters predictions
        - max_users (int) : maximum of users to show on the heatmap
        - max_movies (int) : nb maximal of movies to show on the heatmap
    """
    c=1
    for cluster_id in clustered.group.unique():
        # To improve visibility, we're showing at most max_users users and max_movies movies per cluster.
        # You can change these values to see more users & movies per cluster
        d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
        n_users_in_cluster = d.shape[0]
        
        d = sort_by_rating_density(d, max_movies, max_users)
        
        d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:max_users, :max_movies]
        n_users_in_plot = d.shape[0]
        
        # We're only selecting to show clusters that have more than 9 users, otherwise, they're less interesting
        if len(d) > 9:
            print('cluster # {}'.format(cluster_id))
            print('# of users in cluster: {}.'.format(n_users_in_cluster), '# of users in plot: {}'.format(n_users_in_plot))
            fig = plt.figure(figsize=(15,4))
            ax = plt.gca()

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            labels = d.columns.str[:40]

            ax.set_yticks(np.arange(d.shape[0]) , minor=False)
            ax.set_xticks(np.arange(d.shape[1]) , minor=False)

            ax.set_xticklabels(labels, minor=False)
                        
            ax.get_yaxis().set_visible(False)

            # Heatmap
            heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')

            ax.set_xlabel('movies')
            ax.set_ylabel('User id')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Color bar
            cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
            cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])

            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', labelleft='off') 
            #print('cluster # {} \n(Showing at most {} users and {} movies)'.format(cluster_id, max_users, max_movies))

            plt.show()

def draw_user_embedding(clustered: list,
                        user_id : int,
                        max_users : int,
                        max_movies : int) :
    """
    Draw the cluster heatmap embedding the user.

    Args:
        - clustered (list) : list of the clusters predictions
        - max_users (int) : maximum of users to show on the heatmap
        - max_movies (int) : nb maximal of movies to show on the heatmap
    """
    c=1
    cluster_id = int(clustered[clustered['index']==user_id].group.values)
    # To improve visibility, we're showing at most max_users users and max_movies movies per cluster.
    # You can change these values to see more users & movies per cluster
    d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
    n_users_in_cluster = d.shape[0]
    
    d = sort_by_rating_density(d, max_movies, max_users)
    
    d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
    d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
    d = d.iloc[:max_users, :max_movies]
    n_users_in_plot = d.shape[0]
    
    # We're only selecting to show clusters that have more than 9 users, otherwise, they're less interesting
    #if len(d) > 9:
    print('cluster # {}'.format(cluster_id))
    print('# of users in cluster: {}.'.format(n_users_in_cluster), '# of users in plot: {}'.format(n_users_in_plot))
    fig = plt.figure(figsize=(15,3))
    ax = plt.gca()

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    labels = d.columns.str[:40]

    ax.set_yticks(np.arange(d.shape[0]) , minor=False)
    ax.set_xticks(np.arange(d.shape[1]) , minor=False)

    ax.set_xticklabels(labels, minor=False)
                
    ax.get_yaxis().set_visible(False)

    # Heatmap
    heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')

    ax.set_xlabel('movies')
    ax.set_ylabel('User id')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])

    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', labelleft='off') 
    #print('cluster # {} \n(Showing at most {} users and {} movies)'.format(cluster_id, max_users, max_movies))

    #save file as html
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.savefig('test.png', format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = 'User Embedding' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + f'User id : {user_id} | Cluster id : {cluster_id} |  nb of users in cluster: {n_users_in_cluster}'

    with open(f'./templates/user{user_id}_embedding.html','w') as f:
        f.write(html)
    
    #return path to file
    return(f'user{user_id}_embedding.html')
    
def get_most_rated_movies(user_movie_ratings : pd.DataFrame,
                          max_number_of_movies : int) -> pd.DataFrame :
    """
    Return a dataframe of the most rated movies the count associated.

    Args:
        - user_movie_ratings (pd.DataFrame) : Dataframe of the ratings of the users.
        - max_number_of_movies (int) : max number of movies.

    Returns:
        - most_rated_movies (pd.DataFrame) : Dataframe of the most rated movies and the count associated
    """
    # 1- Count
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
    # 2- sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    # 3- slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies

def get_users_who_rate_the_most(most_rated_movies : pd.DataFrame, 
                                max_number_of_movies : int) -> pd.DataFrame:
    """
    Return a dataframe of the most voting user

    Args:
        - user_movie_ratings (pd.DataFrame) : Dataframe of the ratings of the users.
        - max_number_of_movies (int) : max number of movies.

    Returns:
        - most_rated_movies_users_selection (pd.DataFrame) : Dataframe of the most voting users chosen by user
    """
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    
    return most_rated_movies_users_selection

def sort_by_rating_density(user_movie_ratings : pd.DataFrame,
                           n_movies : int,
                           n_users : int) -> pd.DataFrame :
    """
    Return a dataframe of the most voting user x the most rated movies

    Args:
        - user_movie_ratings (pd.DataFrame) : Dataframe of the ratings of the users.
        - n_movies (int) : max number of movies.
        - n_users (int) : max number of users.

    Returns:
        - most_rated_movies (pd.DataFrame) : Dataframe of the most voting users x most rated movies chosen by user
    """
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies
    
def draw_movies_heatmap(most_rated_movies_users_selection : pd.DataFrame,
                        axis_labels : bool = True):
    """
    Return a heatmap of the most voting user x the most rated movies

    Args :
        - axis_labels (bool) = True : show/not show the label axis
        - most_rated_movies_users_selection (pd.Dataframe) : Dataframe of the most rated movies x the most rating users

    """

    fig = plt.figure(figsize=(15,4))
    ax = plt.gca()
    
    # Draw heatmap
    heatmap = ax.imshow(most_rated_movies_users_selection,  interpolation='nearest', vmin=0, vmax=5, aspect='auto')

    if axis_labels:
        ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]) , minor=False)
        ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]) , minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = most_rated_movies_users_selection.columns.str[:40]
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ax.grid(False)
    ax.set_ylabel('User id')

    # Separate heatmap from color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])

    plt.show()