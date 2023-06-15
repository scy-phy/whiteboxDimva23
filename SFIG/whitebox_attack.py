'''
The code is based on the repository https://github.com/cfeng783/NDSS19_InvariantRuleAD.
we modified the code to perform anomaly detection row by row, instead of performing detection on the entire dataset all together.
The script contains the code to perform the proposed evasion attack
The execution time is slower compared to the AnomlayDetection.py because it executes the attack row by row. 
It can be parallelized for performance by splitting the data and using the 'split' argument. See 'launcher.py'
'''

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from fileinput import filename
from operator import index
from statistics import mean
import time
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from AD import Util
import pickle
import numpy as np

def transform_row_and_attack(df_with_last_2_rows, list_updates, clf, score_threshold, cluster_num, list_float_columns, max_dict, min_dict, cont_vars, entry_list, invar_dict,
                  antecedents_0, consequents_0, antecedents_1, consequents_1, detection_cols, prev_fix, start_time, end_time):
    
    transformed_rows = predicate_generation(df_with_last_2_rows, list_updates, clf, score_threshold, cluster_num)
    transformed_rows = derive_event_driven_predicates(transformed_rows, list_float_columns, max_dict, min_dict, entry_list)
    transformed_rows = transform_event_driven_predicates(transformed_rows, invar_dict, max_dict, min_dict, cont_vars)
    transformed_rows = add_missing_cols(transformed_rows, detection_cols)
    transformed_rows = attack(transformed_rows, df_with_last_2_rows, list_updates, clf, score_threshold, cluster_num, list_float_columns, max_dict, min_dict, cont_vars, entry_list, invar_dict,
                  antecedents_0, consequents_0, antecedents_1, consequents_1, detection_cols, prev_fix, start_time, end_time)

    return transformed_rows

def predicate_generation(transformed_rows, list_updates, gmms, score_threshold, cluster_num):
    'predicate generation'  
    transformed_rows = transformed_rows.fillna(method='ffill')
    cont_vars = []
    for entry in list_updates:
        transformed_rows[entry + '_update'] = transformed_rows[entry].shift(-1) - transformed_rows[entry]  
        cont_vars.append(entry + '_update')

    transformed_rows = transformed_rows[:len(transformed_rows)-1]

    for entry in cont_vars:
        test_X = transformed_rows[entry].values
        test_X = test_X.reshape(-1, 1)
        test_Y = gmms[entry].predict(test_X)
        transformed_rows[entry+'_cluster'] = test_Y
        test_scores = gmms[entry].score_samples(test_X)
        transformed_rows.loc[test_scores<score_threshold[entry].values,entry+'_cluster']=cluster_num[entry].values
        transformed_rows = transformed_rows.drop(entry,axis = 1)
        
    return transformed_rows

def derive_event_driven_predicates(transformed_rows, list_float_columns, max_dict, min_dict, entry_list):
    
    for entry in transformed_rows:
        if entry.endswith('cluster') == True:
            testdf = pd.get_dummies(transformed_rows[entry]).rename(columns=lambda x: entry + '=' + str(x))
            transformed_rows = pd.concat([transformed_rows, testdf], axis=1)
            transformed_rows = transformed_rows.drop(entry, axis = 1)
        else:
            if entry in list_float_columns:
                if min_dict[entry] == max_dict[entry]:
                    transformed_rows = transformed_rows.drop(entry, axis = 1)
                else:
                    transformed_rows[entry]=transformed_rows[entry].apply(lambda x:float(x-min_dict[entry])/float(max_dict[entry]-min_dict[entry]))
            else:
                if entry_list:
                    if entry in entry_list:
                        testdf = pd.get_dummies(transformed_rows[entry]).rename(columns=lambda x: entry + '=' + str(x))
                        transformed_rows = pd.concat([transformed_rows, testdf], axis=1)
                        transformed_rows = transformed_rows.drop(entry, axis = 1)

                    else:
                        transformed_rows[entry + '!=1'] = 1
                        transformed_rows.loc[transformed_rows[entry] == 1, entry + '!=1'] = 0
                        
                        transformed_rows[entry + '!=2'] = 1
                        transformed_rows.loc[transformed_rows[entry] == 2, entry + '!=2'] = 0
                        transformed_rows = transformed_rows.drop(entry, axis = 1)    

                else:
                    transformed_rows[entry + '!=1'] = 1
                    transformed_rows.loc[transformed_rows[entry] == 1, entry + '!=1'] = 0
                    
                    transformed_rows[entry + '!=2'] = 1
                    transformed_rows.loc[transformed_rows[entry] == 2, entry + '!=2'] = 0
                    transformed_rows = transformed_rows.drop(entry, axis = 1)       
    return transformed_rows

def transform_event_driven_predicates(transformed_rows, invar_dict, max_dict, min_dict, cont_vars):
    for target_var in invar_dict:
        icpList = invar_dict[target_var]
        if icpList is not None and len(icpList) > 0:
            icpList.sort()
            if icpList is not None:
                for i in range(len(icpList)+1):
                    if i == 0:
                        invar_entry = Util.conMarginEntry(target_var, icpList[0], 0, max_dict, min_dict)
                        transformed_rows[invar_entry] = 0
                        transformed_rows.loc[transformed_rows[target_var]<icpList[0], invar_entry ] = 1

                    elif i == len(icpList):
                        invar_entry = Util.conMarginEntry(target_var, icpList[i-1], 1,  max_dict, min_dict)
                        transformed_rows[invar_entry] = 0
                        transformed_rows.loc[transformed_rows[target_var]>=icpList[i-1], invar_entry ] = 1
                    else:
                        invar_entry = Util.conRangeEntry(target_var, icpList[i-1],icpList[i], max_dict, min_dict)
                        transformed_rows[invar_entry] = 0
                        transformed_rows.loc[ (transformed_rows[target_var]>=icpList[i-1]) & (transformed_rows[target_var]<=icpList[i]), invar_entry ] = 1

    for var_c in cont_vars:

        transformed_rows = transformed_rows.drop(var_c, axis = 1) 
    return transformed_rows

def add_missing_cols(transformed_rows, detection_cols):
    columns = detection_cols.copy(deep=True)
    for entry in transformed_rows:
        columns[entry] = transformed_rows[entry].values
    return columns

def fix_row(fixes, df, gmms, score_threshold):
    
    for fix in fixes.keys():
        if 'cluster' in fix:
            split = fix.split('_')
            column_diff = df[split[0]].shift(-1) - df[split[0]]
 
            column_diff = column_diff[:len(column_diff)-1]
            test_X = column_diff.values
            test_X = test_X.reshape(-1, 1)
            test_Y = gmms[split[0]+'_update'].predict(test_X)
            score = gmms[split[0]+'_update'].score_samples(test_X) 
            predicted_probs =  gmms[split[0]+'_update'].predict_proba(test_X)
            
            means = gmms[split[0]+'_update'].means_
            distance = column_diff.values - means[fixes[fix]]
            
            while test_Y[0] != fixes[fix] or score < score_threshold[split[0]+'_update'].values:
                df.loc[df.index[0],split[0]]= df.loc[df.index[0],split[0]]+ 0.001*np.sign(distance) 
                column_diff = df[split[0]].shift(-1) - df[split[0]]
                column_diff = column_diff[:len(column_diff)-1]
                test_X = column_diff.values
                test_X = test_X.reshape(-1, 1)
                test_Y = gmms[split[0]+'_update'].predict(test_X)
                score = gmms[split[0]+'_update'].score_samples(test_X) 
        else:
            try:
                df.loc[df.index[0],fix]=int(fixes[fix])
            except ValueError:
                df.loc[df.index[0],fix]=float(fixes[fix])

    return df

def create_dict(attribute, fixes):
    #attribute contains the string form the column with the related condition to 
    #be fulfilled, this function is  parsing the string to recover the value that needs to be set on the 
    #target variable for transformation.
    if 'cluster' in attribute:
        split = attribute.split('_')
        split_value = split[2].split('=')
        fixes[split[0]+'_cluster'] = int(split_value[1])
    else:
        if ('=') in attribute:
            split = attribute.split('=')
        else:
            if '>' in attribute:
                split = attribute.split('>')
            else:
                if '<' in attribute:
                    split = attribute.split('<')
        
        if '!' in split[0]:
            split[0] = split[0].split('!')[0]
            if int(split[1])==2:
                split[1] = 1  
            else: 
                split[1] = 2
       
        fixes[split[0]] = split[1]
    return fixes
    

def attack(transformed_rows, df_with_last_2_rows, list_updates, clf, score_threshold, cluster_num, list_float_columns, max_dict, min_dict, cont_vars, entry_list, invar_dict,
                  antecedents_0, consequents_0, antecedents_1, consequents_1, detectio_cols, prev_fix, start_time, end_time):

    #get columns in the current rows containing zeros
    cols = transformed_rows.columns[transformed_rows.iloc[0]==0]
    
    
    consequents_0['consequent'] = 1
    #check if there is any column that is zero in the row but 1 in the rule(s) and 
    #set the consequent column to 0 in case
    
    consequents_0.loc[(consequents_0[list(cols)]==1).any(1), 'consequent'] = 0
    antecedents_0['antecedent'] = 1
    antecedents_0.loc[(antecedents_0[list(cols)]==1).any(1), 'antecedent'] = 0
    
    #get the rules which are not fulfilled antecedent = 1 and consequent = 0
    #rules contains just the consequents because we need them to fix the row
    indexes = antecedents_0.loc[antecedents_0['antecedent'] == 1]
    rules = consequents_0.iloc[indexes.index].loc[consequents_0['consequent'] == 0]

    consequents_1['consequent'] = 1
    consequents_1.loc[(consequents_1[list(cols)]==1).any(1), 'consequent'] = 0
    antecedents_1['antecedent'] = 1
    antecedents_1.loc[(antecedents_1[list(cols)]==1).any(1), 'antecedent'] = 0

    #get the rules which are not fulfilled antecedent = 1 and consequent = 0
    #rules1 contains just the consequents because we need them to fix the row
    indexes = antecedents_1.loc[antecedents_1['antecedent'] == 1]
    rules1 = consequents_1.iloc[indexes.index].loc[consequents_1['consequent'] == 0]

    fixes = {}
    if not rules.empty or not rules1.empty:
        if not rules.empty:
            #compute bit distance rule(consequent)-row (boolean vector)
            #get columns where the difference = 1 (attributes not fulfilled by the row w.r.t. the rule)
            wrong_attributes = rules[transformed_rows.columns] - transformed_rows.values
            wrong_attributes = wrong_attributes[wrong_attributes==1].dropna(how='all', axis=1).columns        
            for attribute in wrong_attributes:
                #create a dictionary of the desired transformations to fit the rule consequent
                fixes = create_dict(attribute, fixes)
                
            
        if not rules1.empty:
            wrong_attributes1 = rules1[transformed_rows.columns] - transformed_rows.values
            wrong_attributes1 = wrong_attributes1[wrong_attributes1==1].dropna(how='all', axis=1).columns
            for attribute in wrong_attributes1:
                fixes = create_dict(attribute, fixes)
                
        transformed_rows['result'] = 1
    else:
       transformed_rows['result'] = 0
       end_time.append(time.time())


    if transformed_rows.loc[0, 'result'] == 1:
        start_time.append(time.time())
        if prev_fix.keys() == fixes.keys():
            return transformed_rows
        transformed_rows = transformed_rows.drop('result',axis = 1)
        fixed_row = fix_row(fixes, df_with_last_2_rows,clf,score_threshold)
        prev_fix =  fixes
        transformed_rows = transform_row_and_attack(fixed_row, list_updates, clf, score_threshold, cluster_num, list_float_columns,
                                    max_dict, min_dict, cont_vars, entry_list, invar_dict,
                                    antecedents_0, consequents_0, antecedents_1, consequents_1, detectio_cols, prev_fix, start_time, end_time)

    return transformed_rows


def anomaly_detection(transformed_rows,  antecedents_0, consequents_0, antecedents_1, consequents_1):

    cols = transformed_rows.columns[transformed_rows.iloc[0]==0]
    
    consequents_0['consequent'] = 1
    consequents_0.loc[(consequents_0[list(cols)]==1).any(1), 'consequent'] = 0
    antecedents_0['antecedent'] = 1
    antecedents_0.loc[(antecedents_0[list(cols)]==1).any(1), 'antecedent'] = 0

    indexes = antecedents_0.loc[antecedents_0['antecedent'] == 1]
    rules = consequents_0.iloc[indexes.index].loc[consequents_0['consequent'] == 0]

    consequents_1['consequent'] = 1
    consequents_1.loc[(consequents_1[list(cols)]==1).any(1), 'consequent'] = 0
    antecedents_1['antecedent'] = 1
    antecedents_1.loc[(antecedents_1[list(cols)]==1).any(1), 'antecedent'] = 0

    indexes = antecedents_1.loc[antecedents_1['antecedent'] == 1]
    rules1 = consequents_1.iloc[indexes.index].loc[consequents_1['consequent'] == 0]

    if not rules.empty or not rules1.empty:
        transformed_rows['result'] = 1
        if not rules.empty:
            transformed_rows.loc[:, 'count_rules'] = len(rules)

        if not rules1.empty:
            transformed_rows['count_rules1'] = len(rules1)

    else:
       transformed_rows['result'] = 0
    return transformed_rows


def anomaly_detection_slow(df_with_last_2_rows, rule_list_0, rule_list_1,item_dict_0, item_dict_1):

    
    df_with_last_2_rows['result'] = 0
    for rule in rule_list_0:
        if  df_with_last_2_rows['result'].iloc[0] == 1:
            break

        df_with_last_2_rows['antecedent'] = 1
        df_with_last_2_rows['consequent'] = 1

        for item in rule[1]:
            df_with_last_2_rows.loc[df_with_last_2_rows[ item_dict_0[item] ]==0,  'consequent'] = 0
           
            if df_with_last_2_rows['consequent'].iloc[0] == 0:
                    break
                
        if df_with_last_2_rows['consequent'].iloc[0] == 1:
            continue
        else:
            for item in rule[0]:
                df_with_last_2_rows.loc[df_with_last_2_rows[ item_dict_0[item] ]==0,  'antecedent'] = 0

                if df_with_last_2_rows['antecedent'].iloc[0] == 0:
                    break

        
        df_with_last_2_rows.loc[(df_with_last_2_rows[ 'antecedent' ]==1) & (df_with_last_2_rows[ 'consequent' ]==0),  'result'] = 1
            

    for rule in rule_list_1:
        if  df_with_last_2_rows['result'].iloc[0] == 1:
            break
        df_with_last_2_rows['antecedent'] = 1
        df_with_last_2_rows['consequent'] = 1

        for item in rule[1]:
            df_with_last_2_rows.loc[df_with_last_2_rows[ item_dict_1[item] ]==0,  'consequent'] = 0
                    
            if df_with_last_2_rows['consequent'].iloc[0] == 0:
                    break
        
        if df_with_last_2_rows['consequent'].iloc[0] == 1:
            continue
        else:
            for item in rule[0]:
                df_with_last_2_rows.loc[df_with_last_2_rows[ item_dict_1[item] ]==0,  'antecedent'] = 0
                
                if df_with_last_2_rows['antecedent'].iloc[0] == 0:
                    break
            
        df_with_last_2_rows.loc[(df_with_last_2_rows[ 'antecedent' ]==1.0) & (df_with_last_2_rows[ 'consequent' ]==0.0),  'result'] = 1

    return df_with_last_2_rows

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs='+', type=str, default=['./data/SWAT/SWaT_Dataset_Attack_v1.csv'])
    parser.add_argument('-s', '--split', nargs='+', type=str, default=['0'])
    parser.add_argument('-a', '--attack', nargs='+', type=str, default=['NTP'])
    args = parser.parse_args()
    print(args.path)
    print(args.split)
    print(args.attack)

    filename = args.path[0]
    split = args.split[0]
    attack_name = args.attack[0]
    if attack_name == 'NTP':
        WBC_NTP = True
        WBC_NA = False
    elif attack_name == 'NA':
        WBC_NTP = False
        WBC_NA = True
    else:
        raise ValueError("wrong attack argument, allowed NTP or NA")
        
        
    with open('./data/vars/list_updates.pk', 'rb') as fp:
       list_updates = pickle.load(fp)

    antecedents_0 = pd.read_csv('./data/antecedents_0.csv')
    consequents_0 = pd.read_csv('./data/consequents_0.csv')
    antecedents_1 = pd.read_csv('./data/antecedents_1.csv')
    consequents_1 = pd.read_csv('./data/consequents_1.csv')
    with open('./data/vars/gmms.pk', 'rb') as fp:
        gmms = pickle.load(fp)  
    
    score_threshold = pd.read_csv('./data/vars/score_threshold.csv')
    cluster_num = pd.read_csv('./data/vars/cluster_num.csv')

    with open('./data/vars/list_float_columns.pk', 'rb') as fp:
       list_float_columns =  pickle.load(fp)
    with open('./data/vars/min_dict.pk', 'rb') as fp:
        min_dict = pickle.load(fp)
    with open('./data/vars/max_dict.pk', 'rb') as fp:
        max_dict = pickle.load(fp)
    with open('./data/vars/cont_vars.pk', 'rb') as fp:
        cont_vars = pickle.load(fp)
    with open('./data/vars/cont_vars.pk', 'rb') as fp:
        cont_vars = pickle.load(fp)
    with open('./data/vars/entry_list.pk', 'rb') as fp:
        entries_list = pickle.load(fp)
    with open('./data/vars/invar_dict.pk', 'rb') as fp:
        invar_dict = pickle.load(fp)

    test_data = pd.read_csv(filename)
    test_data['actual_ret'] = 0
    try:
        test_data.loc[test_data['NormalAttack']!='Normal', 'actual_ret'] = 1
    except KeyError:
        test_data.loc[test_data['Normal/Attack']!='Normal', 'actual_ret'] = 1
    actual_ret = list(test_data['actual_ret'].values)

    test_data_without_time_and_gt = test_data.drop('Timestamp',axis = 1)
    test_data_without_time_and_gt = test_data_without_time_and_gt.drop('NormalAttack',axis = 1)
    test_data_without_time_and_gt = test_data_without_time_and_gt.drop('actual_ret', axis = 1)

    mod_rows = test_data_without_time_and_gt.copy(deep=True)

    detection_cols = pd.read_csv('./data/detection_cols.csv')
    preds = pd.DataFrame([0]*len(test_data_without_time_and_gt), columns=['result'])
    
    counter = 0
    indices_attack = [i for i, x in enumerate(actual_ret) if x ==1]
    indices_attack = indices_attack[0:-1]
    elapsed = []
    detected = pd.read_csv('./test_data_with_pred.csv')
    detected = detected.loc[detected['result']==1].index.values
    if WBC_NTP:
        for i in indices_attack:
            if counter == 0 or counter%1000 == 0:
                print(str(i)+'/'+str(len(test_data_without_time_and_gt)))
            
            df_with_last_2_rows = pd.DataFrame(test_data_without_time_and_gt.loc[i:i+1])
            prev_fix ={}
            start_time = []
            end_time = []
            transformed_rows = transform_row_and_attack(df_with_last_2_rows, list_updates, gmms, score_threshold, cluster_num, list_float_columns,
                                                max_dict, min_dict, cont_vars, entries_list, invar_dict,  antecedents_0, consequents_0,antecedents_1, 
                                                consequents_1, detection_cols, prev_fix, start_time, end_time)
            try:
                elapsed_recursive = end_time[-1]-start_time[0]
                elapsed.append(elapsed_recursive)
            except IndexError:
                pass
            if counter == 0:
                transformed_rows_df = pd.DataFrame(transformed_rows)
            else:
                transformed_rows_df = transformed_rows_df.append(pd.DataFrame(transformed_rows))
            mod_rows.loc[i]=df_with_last_2_rows.loc[i]
            
            preds['result'].loc[i]=transformed_rows['result'].values
            counter +=1
        
        predict_ret = list(preds['result'].values)
        Util.evaluate_prediction(actual_ret,predict_ret, verbose=1)
        transformed_rows_df.to_csv('./data/whitebox_attack_results_NTP/after_event_whitebox_attack_test.csv', index=False)
        mod_rows.to_csv('./data/whitebox_attack_results_NTP/mod_rows_attack_test.csv', index=False)
        
        pd.DataFrame(elapsed).to_csv('./data/whitebox_attack_results_NTP/elapsed_test.csv')
        
    if WBC_NA:
        for i in detected: 
            if counter ==0 or counter%1000 == 0:
                print(str(i)+'/'+str(len(test_data_without_time_and_gt)))
            
            df_with_last_2_rows = pd.DataFrame(test_data_without_time_and_gt.loc[i:i+1])
            prev_fix ={}
            start_time = []
            end_time = []
            transformed_rows = transform_row_and_attack(df_with_last_2_rows, list_updates, gmms, score_threshold, cluster_num, list_float_columns,
                                                max_dict, min_dict, cont_vars, entries_list, invar_dict,  antecedents_0, consequents_0,antecedents_1, 
                                                consequents_1, detection_cols, prev_fix, start_time, end_time)
            try:
                elapsed_recursive = end_time[-1]-start_time[0]
                elapsed.append(elapsed_recursive)
            except IndexError:
                pass
            if counter == 0:
                transformed_rows_df = pd.DataFrame(transformed_rows)
            else:
                transformed_rows_df= transformed_rows_df.append(pd.DataFrame(transformed_rows))
            mod_rows.loc[i]=df_with_last_2_rows.loc[i]
            
            preds['result'].loc[i]=transformed_rows['result'].values
            counter +=1
    
        
        predict_ret = list(preds['result'].values)
        Util.evaluate_prediction(actual_ret,predict_ret, verbose=1)
        transformed_rows_df.to_csv('./data/whitebox_attack_results_NA/after_event_whitebox_attack_test.csv', index=False)
        mod_rows.to_csv('./data/whitebox_attack_results_NA/mod_rows_attack_test.csv', index=False)
        pd.DataFrame(elapsed).to_csv('./data/whitebox_attack_results_NA/elapsed_test.csv')