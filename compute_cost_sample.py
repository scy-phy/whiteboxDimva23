from numpy import disp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from data_process import DataPreprocess
time = []

#compute euclidean distance and squared difference
def compute_distance(data, adv_data):
    squared_diff = (adv_data-data)**2
    sum_squares = pd.DataFrame.sum(pd.DataFrame(squared_diff), axis=1)
    return np.sqrt(sum_squares),squared_diff

#load data
def get_data_and_calculate(data, adversarial_data_path, columns, compute_hamming):
    adv_data = pd.read_csv(adversarial_data_path)
    adv_data = pd.DataFrame(adv_data, columns=columns)
    if compute_hamming:
        distances, squared_diff = compute_distance(data[columns], adv_data[columns])
        #print(np.mean(distances), np.mean(distances[distances>0]))
        non_zero = squared_diff[columns].ge(0.0001).sum(axis=1)
        print(np.round(np.mean(distances), 3), ' & ',np.round(np.std(distances), 3), ' & ', 
          len(distances[distances>0.00001]), ' & ', np.round(np.mean(non_zero[non_zero>0].values),3),
          ' & ',np.round(np.std(non_zero[non_zero>0].values),3), 
          ' & ',np.round(np.max(non_zero[non_zero>0].values),3), 
          ' & ',np.round(np.min(non_zero[non_zero>0].values),3))
   
    else:
        distances, _ = compute_distance(data[columns], adv_data[columns])
        print(np.round(np.mean(distances), 3), ' & ',np.round(np.std(distances), 3), ' & ', 
          len(distances[distances>0.0001]))
        
#load data for SVM    
def get_data_and_calculate_SVM(data, adv_data, columns):
    
    distances, squared_diff = compute_distance(pd.DataFrame(data), pd.DataFrame(adv_data))

    non_zero = squared_diff.ge(0.0001).sum(axis=1)
    print(np.round(np.mean(distances), 3), ' & ',np.round(np.std(distances), 3), ' & ', 
        len(distances[distances>0.0001]), ' & ', np.round(np.mean(non_zero[non_zero>0].values),3),
        ' & ',np.round(np.std(non_zero[non_zero>0].values),3), 
        ' & ',np.round(np.max(non_zero[non_zero>0].values),3), 
        ' & ',np.round(np.min(non_zero[non_zero>0].values),3))
   
#plot visualization
def plotting(distances, ax,maxlim):
    ax.plot(range(0,len(distances)), distances)
    ax.set_ylim(ymin=0, ymax=maxlim)
    ax.set_xlim(xmin=0, xmax=len(distances))
    

if __name__ == '__main__':
    AR = False
    LTI = False
    PASAD = False
    SFIG= True
    SVM = False
    compute_hamming = LTI or SFIG or SVM
    fig, axs = plt.subplots(6,1)
    if AR:
        data_path = './AR/undersample_test.csv'
        data = pd.read_csv(data_path)
        time = pd.to_datetime(data['DATETIME'], dayfirst=True)
        columns = ['LIT301']
        data = pd.DataFrame(data, columns=columns)
        
        print('AR Model replay')
        adversaria_data_path = './AR/replay_undersample_jan23.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('AR Model random')
        adversaria_data_path = './AR/random_replay_undersample_jan23.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns,compute_hamming)
        
        print('AR Model stale')
        adversaria_data_path = './AR/stale_undersample_jan23.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('AR Model WBC baseline')
        
        adversaria_data_path = './AR/whitebox_attack_all_may22_swat_WBC_baseline.csv'
        columns = ['LIT301']
        elapsed = './AR/elapsed_vector_WBC_baseline.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed)*1000, np.std(elapsed)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('AR Model WBC NTP')
        
        adversaria_data_path = './AR/whitebox_attack_all_may22_swat_WBC_NTP.csv'
        columns = ['LIT301']
        elapsed = './AR/elapsed_vector_WBC_NTP.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed)*1000, np.std(elapsed)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
    
        
        print('AR Model WBC NA')
        
        adversaria_data_path = './AR/whitebox_attack_all_may22_swat_WBC_NA.csv'
        columns = ['LIT301']
        elapsed = './AR/elapsed_vector_WBC_NA.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed)*1000, np.std(elapsed)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
    if LTI:
        data_path = './Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv'
        data = pd.read_csv(data_path)
        time = pd.to_datetime(data['Timestamp'], dayfirst=True)
        columns = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301',
                   'LIT301','AIT401','AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504',
                   'FIT501','FIT502','FIT503','FIT504','PIT501','PIT502','PIT503','FIT601'] 
        data = pd.DataFrame(data, columns=columns)
        
        print('LTI Model replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv'
        
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('LTI Model random replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/random_replay.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('LTI Model stale')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/stale.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('LTI Model WBC baseline')
        
        adversaria_data_path = './LTI/whitebox_attack_all_jan23_swat_WBC_baseline.csv'
        elapsed = './LTI/elapsed_vector_WBC_baseline_jan23.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed>0)*1000, np.std(elapsed>0)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('LTI Model WBC NTP')
        
        adversaria_data_path = './LTI/whitebox_attack_all_jan23_swat_WBC_NTP.csv'
        elapsed = './LTI/elapsed_vector_WBC_NTP_jan23.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed)*1000,' & ', np.std(elapsed)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
    
        
        print('LTI Model WBC NA')
        
        adversaria_data_path = './LTI/whitebox_attack_all_jan23_swat_WBC_NA.csv'
        elapsed = './LTI/elapsed_vector_WBC_NA_jan23.csv'
        elapsed = np.loadtxt(elapsed, delimiter=",")
        print(np.mean(elapsed)*1000,' & ', np.std(elapsed)*1000)
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
    
    if PASAD:
        data_path = './Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv'
        data = pd.read_csv(data_path)
        import re
        time = pd.to_datetime(data['Timestamp'], dayfirst=True)
        columns = ['LIT301']
        #data = min_max_scaler.fit_transform(data[columns])
        data = pd.DataFrame(data, columns=columns)
        
        print('PASAD Model Replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('PASAD Model  Random Replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/random_replay.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('PASAD Model Stale')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/stale.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('PASAD Model WBC baseline')
        adversaria_data_path = './PASAD/whitebox_attack_all_jan23_swat_WBC_baseline.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('PASAD Model WBC NTP')
        adversaria_data_path = './PASAD/whitebox_attack_all_jan23_swat_WBC_NTP.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('PASAD Model WBC NA')
        adversaria_data_path = './PASAD/whitebox_attack_all_jan23_swat_WBC_NA.csv'
        columns = ['LIT301']
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        
    if SFIG:
        data_path = './Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv'
        data = pd.read_csv(data_path)
        time = pd.to_datetime(data['Timestamp'], dayfirst=True)
        columns = ['FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','FIT601','P601','P602','P603'] 
        #data = min_max_scaler.fit_transform(data[columns].values)
        data = pd.DataFrame(data, columns=columns)
        print('SFIG Modelreplay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('SFIG Model random replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/random_replay.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('SFIG Stale')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/stale.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('SFIG Model WBC baseline/NTP')
        adversaria_data_path = './SFIG/data/whitebox_attack_results_NTP/mod_rows_attack_only_attack_rows.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
        
        print('SFIG WBC NA')
        adversaria_data_path = './SFIG/data/whitebox_attack_results_NA/mod_rows_attack_ALL_new_fix_approx.csv'
        get_data_and_calculate(data, adversaria_data_path, columns, compute_hamming)
    
    if SVM:
        data_path = './Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv'
        time = pd.to_datetime(pd.read_csv(data_path)['Timestamp'], dayfirst=True)
        columns = ['LIT101', 'LIT301', 'LIT401']  
        label_col = ['Normal/Attack']
        data = DataPreprocess(data_path, columns, label_col, 1)
        print('SVM model replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv'
        adversaria_data = DataPreprocess(adversaria_data_path, columns, label_col, 1)
        get_data_and_calculate_SVM(data.test_data, adversaria_data.test_data, columns)
        
        print('SVM random replay')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/random_replay.csv'
        adversaria_data = DataPreprocess(adversaria_data_path, columns, label_col, 1)
        get_data_and_calculate_SVM(data.test_data, adversaria_data.test_data, columns)
        
        
        print('SVM stale')
        adversaria_data_path = './Spoofing Framework/SWAT/unconstrained_spoofing/stale.csv'
        adversaria_data = DataPreprocess(adversaria_data_path, columns, label_col, 1)
        get_data_and_calculate_SVM(data.test_data, adversaria_data.test_data, columns)
        
        print('SVM WBC baseline')
        adversarial_data_path = './SVM/adv_examples_baseline_costrained.npy'
        adv_data = np.load(adversarial_data_path, allow_pickle=True)
        get_data_and_calculate_SVM(data.test_data, adv_data, columns)
        
        print('SVM WBC NTP')
        adversarial_data_path = './SVM/adv_examples_ntp_costrained.npy'
        adv_data = np.load(adversarial_data_path, allow_pickle=True)
        get_data_and_calculate_SVM(data.test_data, adv_data, columns)
        
        print('SVM WBC NA')
        adversarial_data_path = './SVM/adv_examples_na_costrained.npy'
        adv_data = np.load(adversarial_data_path, allow_pickle=True)
        get_data_and_calculate_SVM(data.test_data, adv_data, columns)
        
