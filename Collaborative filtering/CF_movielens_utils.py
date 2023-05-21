import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def readdata(dataname, flag_reducedata=False, dataratio=0.3):
    data_path = './'+dataname+'/'
    
    if dataname == 'ml-100k':
        
        #User's data
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(data_path+'u.user', sep='|', names=users_cols, parse_dates=True)
        #Ratings
        rating_cols = ['userId', 'movieId', 'rating', 'time']
        ratings = pd.read_csv(data_path+'u.data', sep='\t', names=rating_cols)
        #Movies
        movie_cols = ['movie_id', 'title', 'release_year', 'video_release_date', 'imdb_url']
        movies = pd.read_csv(data_path+'u.item', sep='|', names=movie_cols, usecols=range(5),encoding='latin-1')
    else:
        ratings = pd.read_csv(data_path+'ratings.csv')
    
    # reducing data
    len_data = ratings.shape[0]
    if flag_reducedata:
        ratings = ratings.iloc[0:int(len_data*dataratio),:].copy()
    else:
        pass
    
    return ratings

def makedata(ratings, flag_test=False):
    rating_table = ratings.pivot_table(values='rating', index='userId', columns='movieId',fill_value=0)
    #rating_table_test = ratings.pivot_table(values='rating', index='user_id', columns='movie_id',fill_value=0)
    
    # test
    if flag_test:
        rating_table = rating_table.iloc[0:20, 0:50]
    else:
        pass
    
    return rating_table

def getsimilarity(rating_table):
    #cos_sim = cosine_similarity([rating_table.iloc[userX,:]], [rating_table.iloc[userY,:]])
    rating_table_sparse = sparse.csr_matrix(rating_table)
    sim = cosine_similarity(rating_table_sparse)
    #sim = cosine_similarity(rating_table)
    
    return sim

def reshapedata(num_user, num_item, rating_table, pred_table):
    # reshape
    rating_table_reshaped = rating_table.reshape(1,num_user*num_item)
    pred_table_reshaped = pred_table.reshape(1,num_user*num_item)
    
    # non-zero
    rating_table_nonzero = np.array([])
    pred_table_nonzero = np.array([])
    
    for i in range(num_user*num_item):
        if rating_table_reshaped[0,i] > 0:
            
            rating_table_nonzero = np.append(rating_table_nonzero, rating_table_reshaped[0,i])
            pred_table_nonzero = np.append(pred_table_nonzero, pred_table_reshaped[0,i])
    
    return rating_table_nonzero, pred_table_nonzero

def visualisedata(rating_table,sim):
    print(f'rating_table.shape: {rating_table.shape}')
    
    num_user = rating_table.shape[0]
    num_item = rating_table.shape[1]
    print(f'num_user: {num_user}, num_item: {num_item}')
    
    # Cosine similarity (User-based)
    print('sim')
    print(sim.shape)
    """
    for i in range(num_user):
        if sum(sim[i,:]) == 0:
            print(f'arg:{i}')"""
    
    return num_user, num_item

def visualiseprediction(rating_table,pred_table):
    print('rating_table:')
    print(rating_table)
    print('prediction:')
    print(pred_table)
    print(f'ratng type: {type(rating_table)}, pred type: {type(pred_table)}')

