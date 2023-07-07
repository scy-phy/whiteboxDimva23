import numpy as np
from whitebox_swat_testbed import *
import joblib
sensors_col = ['LIT101', 'LIT301', 'LIT401']  
label_col = ['Normal/Attack']
train_path="../../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv"
test_path="../../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv"
model_path="./swat_testbed_svm_model_1"
experiments = ['replay', 'random_replay', 'stale','baseline_costrained', 'ntp_costrained', 'na_costrained']#, 'ntp', 'na']

interval = 1

clf = joblib.load(model_path)

#ecml_clf = CClassifierSkLearn(clf)

#metric = CMetricAccuracy()
data = DataPreprocess(train_path, test_path, sensors_col, label_col, interval)
one_class_svm_model_testing(data.test_data, data.test_labels, model_path)
for experiment in experiments:
    print(experiment)
    if experiment in ['replay', 'random_replay', 'stale']:
        adversaria_data_path = '../../Spoofing Framework/SWAT/unconstrained_spoofing/'+experiment+'.csv'
        adversaria_data = DataPreprocess(train_path, adversaria_data_path, sensors_col, label_col, 1)
        adv_examples = adversaria_data.test_data
        y_pred = clf.predict(adversaria_data.test_data)
    else:
        adv_examples = np.load('adv_examples_'+experiment+'.npy', allow_pickle=True)
        y_pred = np.load('adv_examples_pred_'+experiment+'.npy', allow_pickle=True)
        elapsed = np.load('elapsed_'+experiment+'.npy', allow_pickle=True)
        print('MEAN: ',np.round(np.mean(elapsed*1000), 3),'STD: ',np.round(np.std(elapsed*1000), 3))
    
    # y_pred_bef = clf.predict(data.test_data)
    # #y_pred_secml_bef = secml_clf.predict(data.test_data)
    # #y_pred_secml_bef[y_pred_secml_bef == 0] = -1
    # y_pred = clf.predict(adv_examples)
    # #y_pred_secml = secml_clf.predict(adv_examples)
    # acc_before = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_bef))
    # #acc_secml_before = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_secml_bef))
    # acc = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred))
    # #acc_secml = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_secml))

    # # print("Original x0 label: ", y0.item())
    # print("performance by original model before attack on the attack class:", acc_before)
    # #print("performance by secml model before attack on the attack class:", acc_secml_before)
    # print("performance by original model after attack on the attack class:", acc)
    # #print("performance by secml model after attack on the attack class:", acc_secml)
    
    one_class_svm_model_testing(adv_examples, data.test_labels, model_path)