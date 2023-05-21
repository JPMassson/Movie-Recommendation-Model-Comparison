import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, confusion_matrix

from CF_movielens_utils import readdata, makedata, getsimilarity, reshapedata, visualisedata, visualiseprediction

import time
import tqdm

import pdb



def getsimusers(num_user, topN, sim):
    
    for i in range(num_user):
        sim_useri = sim[i,:]
        arg_sorted = np.argsort(-sim_useri)
        arg_sim_users_useri = arg_sorted[1:topN+1]
        
        #print(arg_sim_users_useri)
        
        if i == 0:
            arg_sim_users = arg_sim_users_useri.reshape(1,topN)
        else:
            arg_sim_users = np.append(arg_sim_users,arg_sim_users_useri.reshape(1,topN), axis=0)
        
        #breakpoint()
        
            
    return arg_sim_users

def getpred(num_item, num_user, topN, topN_sim_users, sim, rating_table):
    k = 0
    for userI in tqdm.tqdm(range(num_user)):
        # normalisation
        k = sum([sim[userI,simuser] for simuser in topN_sim_users[userI,:]])
        #print(f'userI: {userI}, k: {k}')
        
        for itemJ in tqdm.tqdm(range(num_item), leave=False):
            np.pi*np.pi
            time.sleep(1/2**4)
            
            pred_userI_itemJ = 0
            
            #pred_userI_itemJ = sum([rating_table.iloc[topN_sim_users[userI,k],itemJ] * sim[userI,topN_sim_users[userI,k]] for k in range(topN)])
            pred_userI_itemJ = sum([rating_table.iloc[topN_sim_users[userI,k],itemJ] * sim[userI,topN_sim_users[userI,k]] for k in range(topN)])/k
            
            if itemJ == 0:
                pred_userI = pred_userI_itemJ
            else:
                pred_userI = np.append(pred_userI,pred_userI_itemJ)
            
        
        if userI == 0:
            pred_table = pred_userI.reshape(1,num_item)
        
        elif userI % 10 == 0:
            pred_table = np.append(pred_table, pred_userI.reshape(1,num_item), axis=0)
            #print(f'{userI} out of {num_user} users done.')
        else:
            pred_table = np.append(pred_table, pred_userI.reshape(1,num_item), axis=0)
    
    return pred_table

# evaluation
def evaluation(rating_table, pred_table):
    print('eval')
    mae_overall = 0
    rmse_overall = 0
    
    mae = 0
    mse = 0
    for i in tqdm.tqdm(range(len(rating_table))):
        np.pi*np.pi
        #print(f'{rating_table[i]} - {pred_table[i]}')
        mae += abs(rating_table[i] - pred_table[i])
        mse += abs(rating_table[i] - pred_table[i]) ** 2
        #breakpoint()
    
    #mae = mean_absolute_error(rating_userI_nonzero, pred_userI_nonzero)
    mae /= len(rating_table)
    
    #rmse = mean_squared_error((rating_userI_nonzero, pred_userI_nonzero))
    mse /= len(rating_table)
    rmse = mse**0.5
    
    mae_overall = mae
    rmse_overall = rmse
    
    print(f'MAE: {mae_overall}, RMSE: {rmse_overall}')

def getconfusion(num_user, num_item, rating_table, pred_table):
    # reshape
    rating_table_reshaped = rating_table.values.reshape(1,num_user*num_item)
    pred_table_reshaped = pred_table.reshape(1,num_user*num_item)
    
    """
    # get GT binaryxx
    rating_table_binary = [1 if rating_table_reshaped[0,i] >= threshold else 0 for i in range(num_user*num_item)]
    pred_table_binary = [1 if pred_table_reshaped[0,i] >= threshold else 0 for i in range(num_user*num_item)]
    """
    
    # non-zero
    rating_table_nonzero = np.array([])
    pred_table_nonzero = np.array([])
    for i in range(num_user*num_item):
        if rating_table_reshaped[0,i] > 0:
            
            #breakpoint()
            
            rating_table_nonzero = np.append(rating_table_nonzero, rating_table_reshaped[0,i])
            pred_table_nonzero = np.append(pred_table_nonzero, pred_table_reshaped[0,i])
    
    rating_table_nonzero = np.ceil(rating_table_nonzero)
    pred_table_nonzero = np.ceil(pred_table_nonzero)
    
    #breakpoint()
    
    confusionmatrix = confusion_matrix(rating_table_nonzero, pred_table_nonzero)
    
    return confusionmatrix

def getbinarycm(num_user, num_item, threshold, rating_table, pred_table):
    # reshape
    rating_table_reshaped = rating_table.values.reshape(1,num_user*num_item)
    pred_table_reshaped = pred_table.reshape(1,num_user*num_item)
    
    #print(f'rating: {rating_table.shape}, pred: {pred_table.shape}')
    #print(f'rating: {rating_table_reshaped.shape}, pred: {pred_table_reshaped.shape}')
    
    # non-zero
    rating_table_nonzero = np.array([])
    pred_table_nonzero = np.array([])
    for i in range(num_user*num_item):
        if rating_table_reshaped[0,i] > 0:
            
            #breakpoint()
            
            rating_table_nonzero = np.append(rating_table_nonzero, rating_table_reshaped[0,i])
            pred_table_nonzero = np.append(pred_table_nonzero, pred_table_reshaped[0,i])
    
    
    rating_table_binary = [1 if rating_table_reshaped[0,i] >= threshold else 0 for i in range(num_user*num_item)]
    pred_table_binary = [1 if pred_table_reshaped[0,i] >= threshold else 0 for i in range(num_user*num_item)]
    
    """
    print(f'rating_nonzero: {len(rating_table_nonzero)}, pred_nonzero: {len(pred_table_nonzero)}')
    print(f'rating: {rating_table_reshaped[0,:25]}')
    print(f'rating_nonzero: {rating_table_nonzero[:25]}')
    print(f'rating_bin: {rating_table_binary[:25]}')
    """
    
    confusionmatrix = confusion_matrix(rating_table_binary, pred_table_binary)
    print(type(confusionmatrix))
    
    recall = confusionmatrix[1,1] / (confusionmatrix[1,1]+confusionmatrix[1,0])
    precision = confusionmatrix[1,1] / (confusionmatrix[1,1]+confusionmatrix[0,1])
    print(f'recall: {recall}, precision: {precision}')
    
    np.savetxt('./output/cmbin_above'+str(threshold)+'.csv', confusionmatrix)
    
    return confusionmatrix

def main():
    
    #dataname = 'ml-100k'
    dataname = 'ml-20m'
    
    flag_reducedata = True
    flag_test = False
    
    dataratio=0.3
    topN = 30
    threshold = 3
    
    # prepare data
    ratings = readdata(dataname,flag_reducedata,dataratio)
    rating_table = makedata(ratings,flag_test)
    
    # Cosine similarity (User-based)
    sim = getsimilarity(rating_table)
    
    num_user, num_item = visualisedata(rating_table,sim)
    
    # get argments of similar user
    topN_sim_users = getsimusers(num_user,topN,sim) # argment of sim users
    
    print('sim users')
    print(topN_sim_users.shape)
    
    # prediction
    pred_table = getpred(num_item,num_user,topN,topN_sim_users,sim,rating_table)
    
    #visualiseprediction(rating_table,pred_table)
    
    # reshape data
    rating_table_nonzero, pred_table_nonzero = reshapedata(num_user,num_item,rating_table.values,pred_table)
    
    # evaluation
    evaluation(rating_table_nonzero, pred_table_nonzero)
    
    cm = getconfusion(num_user, num_item, rating_table, pred_table)
    cm_bin = getbinarycm(num_user, num_item, threshold, rating_table, pred_table)
    
    print('Confusion Matrix')
    print(cm)
    print('Binary Confusion Matrix')
    print(cm_bin)
    #print('done.')
    

if __name__ == '__main__':
    main()
    