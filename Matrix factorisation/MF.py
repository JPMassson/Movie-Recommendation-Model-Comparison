import numpy as np
from numpy import linalg as LA
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, precision_score
from scipy.sparse.linalg import svds

def readData():
    users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=users_cols, encoding='latin-1')
    rating_cols = ['userId', 'movieId', 'rating', 'time']
    ratings = pd.read_csv('u.data', sep='\t', names=rating_cols)
    movie_cols = ['movie_id', 'title', 'release_year', 'video_release_date', 'imdb_url']
    movies = pd.read_csv('u.item', sep='|', names=movie_cols, usecols=range(5) ,encoding='latin-1')


    return ratings


def makeData(ratings):
    matrix_df = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

    return matrix_df

#matrix factorisation using gradient descent

def matrix_factorisation(X, P, Q, K, steps, alpha, beta):
    Q = Q.T
    for step in range(steps):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] > 0 :
                    eij = X[i][j] - np.dot(P[i,:], Q[:,j])

                    sum_of_norms = 0

                    sum_of_norms += LA.norm(P) + LA.norm(Q)

                    eij += ((beta/2) * sum_of_norms)
                    
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))
            error = 0
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if X[i][j] > 0:
                        error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
            if error < 0.001:
                break
    return P, Q.T        


def main():

    ratings = readData()

    ratings_table = makeData(ratings)

    K = 2


    print(ratings_table)

    N = ratings_table.shape[0]

    M = ratings_table.shape[1]

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    steps = 100
    alpha = 0.00002
    beta = float(0.02)

    X = ratings_table.to_numpy()
    estimated_P, estimated_Q = matrix_factorisation(X,P,Q,K,steps,alpha,beta)

    print(N, M)

    modeled_X = np.dot(estimated_P,estimated_Q.T)

    mae = mean_absolute_error(ratings_table, modeled_X)
    print(mae)
    mse = mean_squared_error(ratings_table, modeled_X)
    print(mse)

    print(modeled_X)




if __name__ == '__main__':
    main()