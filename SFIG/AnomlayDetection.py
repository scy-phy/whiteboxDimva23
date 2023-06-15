'''
Created on 12 Jun 2017

@author: cheng_feng
https://github.com/cfeng783/NDSS19_InvariantRuleAD/blob/2885c38f4c404f25fe55117c42ce4f7b9637783b/InvarintRuleAD/AD/AnomlayDetection.py

modified by Alessandro Erba, the script now supports storing of the detector parameters, for testing without re-training the model.
The 'load_from_file' variable True will perform detection based on stored detector files, if False training, testing and storing of detector files is performed.

The current version of the script with the 'load_from_file' set to true will reproduce the results of Table 6 in the paper

'''
import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.linear_model import Lasso
from sklearn import metrics
from AD import Util
import time
import pickle 

DATASET='SWAT'

'parameters to tune'
'best parameters swat'
if DATASET == 'SWAT':
    eps = 0.01 #same as in the paper
    sigma = 1.1 #buffer scaler
    theta_value = 0.08 #0.1 same as in the paper
    gamma_value = 0.9 #same as in the paper
    max_k=4

'wadi '
if DATASET == 'WADI':
    eps = 0.001 #same as in the paper
    sigma = 1.1 #buffer scaler
    theta_value = 0.16 #0.1 same as in the paper
    gamma_value = 0.9 #same as in the paper
    max_k=4

load_from_file=True

if not(load_from_file):
    'data preprocessing'
    if DATASET == 'SWAT':
        training_data = pd.read_csv("../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv")
        test_data = pd.read_csv("../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv") #change with the data we need to test with

        training_data = training_data.drop('Timestamp',1)
        training_data = training_data.drop('NormalAttack',1)
        
    if DATASET == 'WADI':
        training_data = pd.read_csv("./data/WADI/14_days_clean.csv")
        test_data = pd.read_csv("./data/WADI/attacks_october_clean_with_label.csv")
        training_data = training_data.drop(['Row', 'Date', 'Time'],1)
        test_data = test_data.drop(['Row'],1)
        
    test_data_attacks = []#test_data_attack_stale_constr, test_data_attack_replay_constr, test_data_attack_gaussian_constr]#, test_data_attack_constant, test_data_attack_gaussian]#, test_data_attack_gaussian_2]
    
    'predicate generation'  
    cont_vars = []
    training_data = training_data.fillna(method='ffill')
    test_data = test_data.fillna(method='ffill')
    list_updates = []
    for i in range(0,len(test_data_attacks)):
        test_data_attacks[i] = test_data_attacks[i].fillna(method='ffill')

    for entry in training_data:
        if training_data[entry].dtypes == np.float64:
            max_value = training_data[entry].max()
            min_value = training_data[entry].min()
            if max_value != min_value:
                training_data[entry + '_update'] = training_data[entry].shift(-1) - training_data[entry]
                test_data[entry + '_update'] = test_data[entry].shift(-1) - test_data[entry]
                list_updates.append(entry)
                for i in range(0,len(test_data_attacks)):
                    test_data_attacks[i][entry + '_update'] = test_data_attacks[i][entry].shift(-1) - test_data_attacks[i][entry]      
                
                cont_vars.append(entry + '_update')
        
    training_data = training_data[:len(training_data)-1]
    test_data = test_data[:len(test_data)-1]
    for i in range(0,len(test_data_attacks)):
        test_data_attacks[i]= test_data_attacks[i][:len(test_data_attack_replay_constr)-1]

    anomaly_entries = []
    anomaly_entries_attacks = [[],[],[],[]]
    row = [[0]*len(cont_vars)]
    score_thresholds = pd.DataFrame(row, columns=cont_vars)
    cluster_nums = pd.DataFrame(row, columns=cont_vars)
    gmms ={}
    for entry in cont_vars:
        print('generate distribution-driven predicates for',entry)
        X = training_data[entry].values
        X = X.reshape(-1, 1)
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 6)
        cluster_num = 0
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                clf = gmm
                cluster_num = n_components
            
        gmms[entry]=clf   
        Y = clf.predict(X)
        training_data[entry+'_cluster'] = Y
        cluster_num = len(training_data[entry+'_cluster'].unique() )
        cluster_nums[entry]= cluster_num

        scores = clf.score_samples(X)
        score_threshold = scores.min()*sigma

        score_thresholds[entry] = score_threshold

            
        test_X = test_data[entry].values
        test_X = test_X.reshape(-1, 1)
        test_Y = clf.predict(test_X)
        test_data[entry+'_cluster'] = test_Y
        test_scores = clf.score_samples(test_X)
        test_data.loc[test_scores<score_threshold,entry+'_cluster']=cluster_num
        if len(test_data.loc[test_data[entry+'_cluster']==cluster_num,:])>0:
            anomaly_entry = entry+'_cluster='+str(cluster_num)
            anomaly_entries.append(anomaly_entry)
        
        for i in range(0,len(test_data_attacks)):
            test_X = test_data_attacks[i][entry].values
            test_X = test_X.reshape(-1, 1)
            test_Y = clf.predict(test_X)
            test_data_attacks[i][entry+'_cluster'] = test_Y
            test_scores = clf.score_samples(test_X)
            test_data_attacks[i].loc[test_scores<score_threshold,entry+'_cluster']=cluster_num
            if len(test_data_attacks[i].loc[test_data_attacks[i][entry+'_cluster']==cluster_num,:])>0:
                anomaly_entry_attack = entry+'_cluster='+str(cluster_num)
                anomaly_entries_attacks[i].append(anomaly_entry_attack)
        
            
        training_data = training_data.drop(entry,1)
        test_data = test_data.drop(entry,1)
        for i in range(0,len(test_data_attacks)):
            test_data_attacks[i] = test_data_attacks[i].drop(entry,1)
        
    'save intermediate result'    

    cluster_nums.to_csv('./data/vars/cluster_num.csv', index =False) 
    score_thresholds.to_csv('./data/vars/score_threshold.csv', index =False)
    training_data.to_csv("./data/swat_after_distribution_normal.csv", index=False)
    test_data.to_csv("./data/swat_after_distribution_attack.csv", index=False)
    for i in range(0,len(test_data_attacks)):
        test_data_attacks[i].to_csv("./data/swat_after_distribution_spoofing_attack"+str(i)+".csv", index=False)
    # training_data = pd.read_csv("../../data/swat_after_distribution_normal.csv")
    # test_data = pd.read_csv("../../data/swat_after_distribution_attack.csv")
        
    'derive event driven predicates'
    cont_vars = []
    disc_vars = []
        
    max_dict = {}
    min_dict = {}
        
    onehot_entries = {}
    dead_entries = []
    list_float_columns = []
    entries_list = []
    for entry in training_data:
        if entry.endswith('cluster') == True:
            newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            if len( newdf.columns.values.tolist() ) <= 1:
                unique_value = training_data[entry].unique()[0]
                dead_entries.append(entry + '=' + str(unique_value))
                training_data = pd.concat([training_data, newdf], axis=1)
                training_data = training_data.drop(entry, 1)
                
                testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                test_data = pd.concat([test_data, testdf], axis=1)
                test_data = test_data.drop(entry, 1)
                for i in range(0,len(test_data_attacks)):
                    testdf_attack = pd.get_dummies(test_data_attacks[i][entry]).rename(columns=lambda x: entry + '=' + str(x))
                    test_data_attacks[i] = pd.concat([test_data_attacks[i], testdf_attack], axis=1)
                    test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
            else:
                onehot_entries[entry]= newdf.columns.values.tolist()
                training_data = pd.concat([training_data, newdf], axis=1)
                training_data = training_data.drop(entry, 1)
                
                testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                test_data = pd.concat([test_data, testdf], axis=1)
                test_data = test_data.drop(entry, 1)
                for i in range(0,len(test_data_attacks)):
                    testdf_attack = pd.get_dummies(test_data_attacks[i][entry]).rename(columns=lambda x: entry + '=' + str(x))
                    test_data_attacks[i] = pd.concat([test_data_attacks[i], testdf_attack], axis=1)
                    test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
        else:
            if training_data[entry].dtypes == np.float64:
                max_value = training_data[entry].max()
                min_value = training_data[entry].min()
                list_float_columns.append(entry)
                if max_value == min_value:
                    training_data = training_data.drop(entry, 1)
                    test_data = test_data.drop(entry, 1)
                    for i in range(0,len(test_data_attacks)):
                        test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
                else:
                    training_data[entry]=training_data[entry].apply(lambda x:float(x-min_value)/float(max_value-min_value))
                    cont_vars.append(entry)
                    max_dict[entry] = max_value
                    min_dict[entry] = min_value
                    test_data[entry]=test_data[entry].apply(lambda x:float(x-min_value)/float(max_value-min_value))
                    for i in range(0,len(test_data_attacks)):
                        test_data_attacks[i][entry]=test_data_attacks[i][entry].apply(lambda x:float(x-min_value)/float(max_value-min_value))
            else:
                newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                if len( newdf.columns.values.tolist() ) <= 1:
                    entries_list.append(entry)
                    unique_value = training_data[entry].unique()[0]
                    dead_entries.append(entry + '=' + str(unique_value))
                    training_data = pd.concat([training_data, newdf], axis=1)
                    training_data = training_data.drop(entry, 1)
                        
                    testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                    test_data = pd.concat([test_data, testdf], axis=1)
                    test_data = test_data.drop(entry, 1)
                    for i in range(0,len(test_data_attacks)):
                        testdf_attack = pd.get_dummies(test_data_attacks[i][entry]).rename(columns=lambda x: entry + '=' + str(x))
                        test_data_attacks[i] = pd.concat([test_data_attacks[i], testdf_attack], axis=1)
                        test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
                        
                elif len( newdf.columns.values.tolist() ) == 2:
                    entries_list.append(entry)
                    disc_vars.append(entry)
                    training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + training_data[entry].astype(int).astype(str)
                    onehot_entries[entry]= newdf.columns.values.tolist()
                    training_data = pd.concat([training_data, newdf], axis=1)
                    training_data = training_data.drop(entry, 1)
                        
                    testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                    test_data = pd.concat([test_data, testdf], axis=1)
                    test_data = test_data.drop(entry, 1)
                    
                    for i in range(0,len(test_data_attacks)):
                        testdf_attack = pd.get_dummies(test_data_attacks[i][entry]).rename(columns=lambda x: entry + '=' + str(x))
                        test_data_attacks[i] = pd.concat([test_data_attacks[i], testdf_attack], axis=1)
                        test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
                    
                else:
                    disc_vars.append(entry)
                    training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + training_data[entry].astype(int).astype(str)
                        
                    training_data[entry + '!=1'] = 1
                    training_data.loc[training_data[entry] == 1, entry + '!=1'] = 0
                        
                    training_data[entry + '!=2'] = 1
                    training_data.loc[training_data[entry] == 2, entry + '!=2'] = 0
                    training_data = training_data.drop(entry, 1)
                        
                    test_data[entry + '!=1'] = 1
                    test_data.loc[test_data[entry] == 1, entry + '!=1'] = 0
                        
                    test_data[entry + '!=2'] = 1
                    test_data.loc[test_data[entry] == 2, entry + '!=2'] = 0
                    test_data = test_data.drop(entry, 1)    
                    for i in range(0,len(test_data_attacks)):
                        test_data_attacks[i][entry + '!=1'] = 1
                        test_data_attacks[i].loc[test_data_attacks[i][entry] == 1, entry + '!=1'] = 0
                            
                        test_data_attacks[i][entry + '!=2'] = 1
                        test_data_attacks[i].loc[test_data_attacks[i][entry] == 2, entry + '!=2'] = 0
                        test_data_attacks[i] = test_data_attacks[i].drop(entry, 1)
                
    
    invar_dict = {}
    for entry in disc_vars:  
        print('generate event-driven predicates for',entry)
        for roundi in [0, 1]:
            print( 'round: ' + str(roundi) )
            tempt_data = training_data.copy()
            tempt_data[entry] = 0
            if roundi == 0:
                tempt_data.loc[(tempt_data[entry+'_shift']=='1->0') | (tempt_data[entry+'_shift']=='1->2') | (tempt_data[entry+'_shift']=='0->2'), entry] = 99
            if roundi == 1:
                tempt_data.loc[(tempt_data[entry+'_shift']=='2->0') | (tempt_data[entry+'_shift']=='2->1') | (tempt_data[entry+'_shift']=='0->1'), entry] = 99
                
            for target_var in cont_vars:    
                active_vars = list(cont_vars)
                active_vars.remove(target_var)
                    
                X = tempt_data.loc[tempt_data[entry]==99, active_vars].values
                Y = tempt_data.loc[tempt_data[entry]==99, target_var].values
                    
                X_test = tempt_data[active_vars].values.astype(np.float)
                Y_test = tempt_data[target_var].values.astype(np.float)
                    
                if len(Y)>5:
                    lgRegr = Lasso(alpha=1, normalize=False)
                        
                    lgRegr.fit(X, Y)
                    y_pred = lgRegr.predict(X)
                        
                    mae = metrics.mean_absolute_error(Y, y_pred)
                    dist = list(np.array(Y) - np.array(y_pred))
                    dist = map(abs, dist)
                    max_error = max(dist)
                    mae_test = metrics.mean_absolute_error(Y_test, lgRegr.predict( X_test ))
                        
                    min_value = tempt_data.loc[tempt_data[entry]==99, target_var].min()
                    max_value = tempt_data.loc[tempt_data[entry]==99, target_var].max()
    #                 print(target_var,max_error)  
                    if max_error < eps:
                        max_error = max_error*sigma
                        must = False
                        for coef in lgRegr.coef_:
                            if coef > 0:
                                must = True
                        if must == True:
                            invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_-max_error, '<', max_dict, min_dict, lgRegr.coef_, active_vars)
                            training_data[invar_entry] = 0
                            training_data.loc[training_data[target_var]< lgRegr.intercept_-max_error, invar_entry ] = 1
                                
                            invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_+max_error, '>', max_dict, min_dict, lgRegr.coef_, active_vars)
                            training_data[invar_entry] = 0
                            training_data.loc[training_data[target_var] > lgRegr.intercept_+max_error, invar_entry ] = 1
                        else:
                            if target_var not in invar_dict:
                                invar_dict[target_var] = []
                            icpList = invar_dict[target_var]
                                
                            if lgRegr.intercept_-max_error > 0  and lgRegr.intercept_-max_error <1:
                                invar_dict[target_var].append(lgRegr.intercept_-max_error)
                                
                            if lgRegr.intercept_+max_error > 0 and lgRegr.intercept_+max_error <1:
                                invar_dict[target_var].append(lgRegr.intercept_+max_error)
        training_data = training_data.drop(entry+'_shift',1)
            
    for target_var in invar_dict:
        icpList = invar_dict[target_var]
        if icpList is not None and len(icpList) > 0:
            icpList.sort()
            if icpList is not None:
                for i in range(len(icpList)+1):
                    if i == 0:
                        invar_entry = Util.conMarginEntry(target_var, icpList[0], 0, max_dict, min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[ training_data[target_var]<icpList[0], invar_entry ] = 1
                            
                        test_data[invar_entry] = 0
                        test_data.loc[ test_data[target_var]<icpList[0], invar_entry ] = 1
                        for j in range(0,len(test_data_attacks)):
                            test_data_attacks[j][invar_entry] = 0
                            test_data_attacks[j].loc[ test_data_attacks[j][target_var]<icpList[0], invar_entry ] = 1
                        
                        
                            
                    elif i == len(icpList):
                        invar_entry = Util.conMarginEntry(target_var, icpList[i-1], 1,  max_dict, min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[ training_data[target_var]>=icpList[i-1], invar_entry ] = 1
                            
                        test_data[invar_entry] = 0
                        test_data.loc[ test_data[target_var]>=icpList[i-1], invar_entry ] = 1
                        for j in range(0,len(test_data_attacks)):
                            test_data_attacks[j][invar_entry] = 0
                            test_data_attacks[j].loc[ test_data_attacks[j][target_var]>=icpList[i-1], invar_entry ] = 1
                            
                    else:
                        invar_entry = Util.conRangeEntry(target_var, icpList[i-1],icpList[i], max_dict, min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[ (training_data[target_var]>=icpList[i-1]) & (training_data[target_var]<=icpList[i]), invar_entry ] = 1
                            
                        test_data[invar_entry] = 0
                        test_data.loc[ (test_data[target_var]>=icpList[i-1]) & (test_data[target_var]<=icpList[i]), invar_entry ] = 1
                        for j in range(0,len(test_data_attacks)):
                            test_data_attacks[j][invar_entry] = 0
                            test_data_attacks[j].loc[ (test_data_attacks[j][target_var]>=icpList[i-1]) & (test_data_attacks[j][target_var]<=icpList[i]), invar_entry ] = 1
                        
    for var_c in cont_vars:
        training_data = training_data.drop(var_c,1)
        test_data = test_data.drop(var_c,1) 
        for i in range(0,len(test_data_attacks)):
            test_data_attacks[i] = test_data_attacks[i].drop(var_c,1) 

    'save intermediate result'
    training_data.to_csv("./data/after_event_normal.csv", index=False)
    test_data.to_csv("./data/after_event_attack.csv", index=False)
    for i in range(0,len(test_data_attacks)):
        test_data_attacks[i].to_csv("./data/after_event_spoofing_attack"+str(i)+".csv", index=False)

    with open('./data/entries/anomaly_entries.pk', 'wb') as fp:
        pickle.dump(anomaly_entries, fp)

    e=0
    for entry in anomaly_entries_attacks:
        with open('./data/entries/anomaly_entries'+str(e)+'.pk', 'wb') as fp:
            pickle.dump(entry, fp)
        e = e + 1

    with open('./data/entries/dead_entries.pk', 'wb') as fp:
        pickle.dump(dead_entries, fp)
        


   # print('anomaly entries')
  #  print(anomaly_entries)
   # print(anomaly_entries_attacks)
   # print('dead entries')
   # print(dead_entries)
    'Rule mining'
    #SWaT key array
    keyArray = [['FIT101','LIT101','MV101','P101','P102'], ['AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206'],
            ['DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302'], ['AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401'],
            ['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503'],['FIT601','P601','P602','P603']]


    print('Start rule mining')
    print('Gamma=' + str(gamma_value) + ', theta=' + str(theta_value))
    start_time = time.time()
    rule_list_0, item_dict_0 = Util.getRules(training_data, dead_entries, keyArray, mode=0, gamma=gamma_value, max_k=max_k, theta=theta_value)
    print('finish mode 0')
    ##mode 2 is quite costly, use mode 1 if want to save time
    rule_list_1, item_dict_1 = Util.getRules(training_data, dead_entries, keyArray, mode=1, gamma=gamma_value, max_k=max_k, theta=theta_value)
    print('finish mode 1')
    end_time = time.time()
    time_cost = (end_time-start_time)*1.0/60
    print('rule mining time cost: ' + str(time_cost))
    
    rules = []
    for rule in rule_list_1:
        valid = False
        for item in rule[0]:
            if 'cluster' in item_dict_1[item]:
                valid = True
                break
        if valid == False:
            for item in rule[1]:
                if 'cluster' in item_dict_1[item]:
                    valid = True
                    break
        if valid == True:
            rules.append(rule)
    rule_list_1 = rules
    print('rule count: ' +str(len(rule_list_0) + len(rule_list_1)))
    
    with open('./data/rules/rule_list_0.pk', 'wb') as fp:
        pickle.dump(rule_list_0, fp)   
    with open('./data/rules/rule_list_1.pk', 'wb') as fp:
        pickle.dump(rule_list_1, fp)
    with open('./data/rules/item_dict_0.pk', 'wb') as fp:
        pickle.dump(item_dict_0, fp)   
    with open('./data/rules/item_dict_1.pk', 'wb') as fp:
        pickle.dump(item_dict_1, fp)
    with open('./data/vars/list_updates.pk', 'wb') as fp:
        pickle.dump(list_updates, fp)
    with open('./data/vars/gmms.pk', 'wb') as fp:
        pickle.dump(gmms, fp)      
 
    with open('./data/vars/list_float_columns.pk', 'wb') as fp:
        pickle.dump(list_float_columns, fp)
    with open('./data/vars/min_dict.pk', 'wb') as fp:
        pickle.dump(min_dict, fp)
    with open('./data/vars/max_dict.pk', 'wb') as fp:
        pickle.dump(max_dict, fp)
    with open('./data/vars/cont_vars.pk', 'wb') as fp:
        pickle.dump(cont_vars, fp)
    with open('./data/vars/cont_vars.pk', 'wb') as fp:
        pickle.dump(cont_vars, fp)
    with open('./data/vars/entry_list.pk', 'wb') as fp:
        pickle.dump(entries_list, fp)
    with open('./data/vars/invar_dict.pk', 'wb') as fp:
        pickle.dump(invar_dict, fp)
    

    ' arrange rules according to phase '
    phase_dict = {}
    for i in range(1,len(keyArray)+1):
        phase_dict[i] = []
    
    for rule in rule_list_0:
        strPrint = ''
        first = True
        for item in rule[0]:
            strPrint += item_dict_0[item] + ' and '
            if first == True:
                first = False
                for i in range(0,len(keyArray)):
                    for key in keyArray[i]:
                        if key in item_dict_0[item]:
                            phase = i+1
                            break
                        
        strPrint = strPrint[0:len(strPrint)-4] 
        strPrint += '---> '
        for item in rule[1]:
            strPrint += item_dict_0[item] + ' and '
        strPrint = strPrint[0:len(strPrint)-4]
        phase_dict[phase].append(strPrint)
    
    for rule in rule_list_1:
        strPrint = ''
        first = True
        for item in rule[0]:
            strPrint += item_dict_1[item] + ' and '
            if first == True:
                first = False
                for i in range(0,6):
                    for key in keyArray[i]:
                        if key in item_dict_1[item]:
                            phase = i+1
                            break
                            
        strPrint = strPrint[0:len(strPrint)-4] 
        strPrint += '---> '
        for item in rule[1]:
            strPrint += item_dict_1[item] + ' and '
        strPrint = strPrint[0:len(strPrint)-4]
        phase_dict[phase].append(strPrint)


        
    # print ' print rules'
    invariance_file = "./data/invariants/invariants_gamma=" + str(gamma_value)+'&theta=' + str(theta_value) + ".txt"
    with open(invariance_file, "w") as myfile:
        for i in range(1,len(keyArray)+1):
            myfile.write('P' + str(i) + ':' + '\n')
            
            
            for rule in phase_dict[i]:
                myfile.write(rule + '\n')
                myfile.write('\n')
            
            myfile.write('--------------------------------------------------------------------------- '+'\n') 
        myfile.close()
    
if load_from_file:
    with open ('./data/entries/anomaly_entries.pk', 'rb') as fp:
        anomaly_entries = pickle.load(fp)
    anomaly_entries_attacks = []
    for i in range(0,1):
        with open ('./data/entries/anomaly_entries'+str(i)+'.pk', 'rb') as fp:
            anomaly_entries_attacks.append(pickle.load(fp))

    with open ('./data/entries/dead_entries.pk', 'rb') as fp:
        dead_entries = pickle.load(fp)   

    training_data = pd.read_csv("./data/after_event_normal.csv")
    test_data = pd.read_csv("./data/after_event_attack.csv")
    #test_data = pd.read_csv("./data/whitebox_attack_results_ALL/after_event_whitebox_attack_ALL_new.csv")
    
    test_data_attacks = []
    #for i in range(0,1):
    test_data_attacks.append(pd.read_csv("./data/SWAT/unconstrained_spoofing/after_event_replay.csv"))#/after_event_spoofing_attack"+str(i)+".csv"))
    test_data_attacks.append(pd.read_csv("./data/SWAT/unconstrained_spoofing/after_event_random_replay.csv"))
    test_data_attacks.append(pd.read_csv("./data/SWAT/unconstrained_spoofing/after_event_stale.csv"))
    
    test_data_attacks.append(pd.read_csv("./data/whitebox_attack_results_NTP/mod_rows_attack_only_attack_rows.csv"))
    test_data_attacks.append(pd.read_csv("./data/whitebox_attack_results_NA/after_event_whitebox_attack_ALL_new.csv"))
      
    
    with open('./data/rules/rule_list_0.pk', 'rb') as fp:
        rule_list_0 = pickle.load(fp)   
    with open('./data/rules/rule_list_1.pk', 'rb') as fp:
       rule_list_1 = pickle.load(fp)    
    with open('./data/rules/item_dict_0.pk', 'rb') as fp:
        item_dict_0 = pickle.load(fp)   
    with open('./data/rules/item_dict_1.pk', 'rb') as fp:
       item_dict_1 = pickle.load(fp)    
    
    
test_data_attacks.insert(0, test_data)
test_datasets = test_data_attacks
anomaly_entries_attacks.insert(0,anomaly_entries)
anomaly_entries_list = anomaly_entries_attacks
attack_names = ['original', 'replay', 'random replay', 'stale', 'WBC NTP', 'WBC NA']
###### use the invariants to do anomaly detection
print('start classification')
i = 0
for test in test_datasets:
    if load_from_file:
        print(attack_names[i])
    test['result'] = 0
    #for entry in anomaly_entries_list[i]:
    #    test.loc[test[entry]==1,  'result'] = 1

    test['actual_ret'] = 0
    if DATASET=='WADI':
        test.loc[test['ATT_FLAG']!='False', 'actual_ret'] = 1
        
    if DATASET == 'SWAT':    
        try:
            test.loc[test['NormalAttack']!='Normal', 'actual_ret'] = 1
        except KeyError:
            try:
                test.loc[test['Normal/Attack']!='Normal', 'actual_ret'] = 1
            except KeyError:
                test['actual_ret'] = test_datasets[0]['actual_ret']
    actual_ret = list(test['actual_ret'].values)

    start_time = time.time()
    num = 0
    for rule in rule_list_0:
        num += 1
        test.loc[:,'antecedent'] = 1
        test.loc[:,'consequent'] = 1
        strPrint = ' '
        for item in rule[0]:
            if item_dict_0[item] in test:
                test.loc[test[ item_dict_0[item] ]==0,  'antecedent'] = 0
            else:
                test.loc[:,  'antecedent'] = 0
            strPrint += str(item_dict_0[item]) + ' '
        strPrint += '-->'
        for item in rule[1]:
            if item_dict_0[item] in test:
                test.loc[test[ item_dict_0[item] ]==0,  'consequent'] = 0
            else:
                test.loc[:,  'consequent'] = 0
            strPrint += ' ' + str(item_dict_0[item])
        test.loc[(test[ 'antecedent' ]==1) & (test[ 'consequent' ]==0),  'result'] = 1
        

    for rule in rule_list_1:
        num += 1
        test.loc[:,'antecedent'] = 1
        test.loc[:,'consequent'] = 1
        strPrint = ' '
        for item in rule[0]:
            if item_dict_1[item] in test:
                test.loc[test[ item_dict_1[item] ]==0,  'antecedent'] = 0
            else:
                test.loc[:,  'antecedent'] = 0
            strPrint += str(item_dict_1[item]) + ' '
    
        strPrint += '-->'
        
        for item in rule[1]:
            if item_dict_1[item] in test:
                test.loc[test[ item_dict_1[item] ]==0,  'consequent'] = 0
            else:
                test.loc[:,  'consequent'] = 0
            strPrint += ' ' + str(item_dict_1[item]) 
        test.loc[(test[ 'antecedent' ]==1) & (test[ 'consequent' ]==0),  'result'] = 1
        
    end_time = time.time()
    time_cost = (end_time-start_time)*1.0/60
    print( 'detection time cost: ' + str(time_cost))
    predict_ret = list(test['result'].values)

    Util.evaluate_prediction(actual_ret,predict_ret, verbose=1)
    i= i + 1

    #test_data.to_csv('./data/test_data_with_pred.csv', index=False)