import pandas as pd
import numpy as np
np.random.seed(seed=123)
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
from tqdm import tqdm

from secml.ml.classifiers import CClassifierSkLearn
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray

import time
import warnings
warnings.filterwarnings("ignore")

class DataPreprocess:
    def __init__(self, train_data_path, test_data_path, sensor_cols, label_col, interval):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.sensor_cols = sensor_cols
        self.label_col = label_col
        self.interval =  interval
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.load_data()
        self.train_data, self.train_labels = self.data_process(self.train_data, self.train_labels, self.interval)
        self.test_data, self.test_labels = self.data_process(self.test_data, self.test_labels, self.interval)
        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
        
        indices = np.random.choice(len(self.train_data), np.int(np.round(0.2*len(self.train_data))), replace=False)
        self.train_data = np.array(self.train_data)[indices]
        self.train_labels = np.array(self.train_labels)[indices]
        
    def load_data(self):
        train_data = pd.read_csv(self.train_data_path)
        train_label = train_data[self.label_col].to_numpy()
        train_data = train_data[self.sensor_cols].to_numpy()
        test_data = pd.read_csv(self.test_data_path)
        test_label = test_data[self.label_col].to_numpy()
        test_data = test_data[self.sensor_cols].to_numpy()
        return train_data, train_label, test_data, test_label
         
    def data_process(self, data_set, label_set, interval):
        data=[]
        labels = []
        for i in range(0,len(data_set)-interval):
            vector=data_set[i].tolist()+data_set[i+interval].tolist()
            label = 1 if (label_set[i] == 'Normal') and (label_set[i+interval] == 'Normal') else -1
            #print(label)
            data.append(vector)
            labels.append(label)
        return np.asarray(data), np.asarray(labels)
        
def one_class_svm_model_training(data, model_path):
    clf =  svm.OneClassSVM(kernel='linear', gamma=0.01, nu=0.02).fit(data.train_data)
    prediction = clf.predict(data.train_data)
    print(accuracy_score(data.train_labels, prediction))
    joblib.dump(clf, model_path)

def one_class_svm_model_tuning(data):
    tuned_parameters = [{'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 0.5, 1, 1.5, 2], 
                        'nu': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.5, 0.99]}]

    scores = ['recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            svm.OneClassSVM(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(data.train_data, data.train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = data.test_labels, clf.predict(data.test_data)
        print(classification_report(y_true, y_pred))
        accuracy=accuracy_score(y_true, y_pred)
        f1=f1_score(y_true, y_pred, pos_label=-1)
        recall=recall_score(y_true, y_pred, pos_label=-1)
        precision=precision_score(y_true, y_pred,  pos_label=-1)
        tp,fn,fp,tn = confusion_matrix(y_true, y_pred).ravel() #labels are inverted 1 negative class
        fpr=fp/(fp+tn)
        print('TP:{:.2f},FN:{:.2f},FP:{:.2f},TN:{:.2f}'.format(tp,fn,fp,tn))
        print('& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(accuracy,f1,precision,recall,fpr))
        print()

    
def one_class_svm_model_testing(test_data, test_labels, model_path):
    clf=joblib.load(model_path)
    prediction = clf.predict(test_data)
    accuracy=accuracy_score(test_labels, prediction)
    f1=f1_score(test_labels, prediction, pos_label=-1)
    recall=recall_score(test_labels, prediction, pos_label=-1)
    precision=precision_score(test_labels, prediction,  pos_label=-1)
    tp,fn,fp,tn = confusion_matrix(test_labels, prediction).ravel() #labels are inverted 1 negative class
    fpr=fp/(fp+tn)
    print('TP:{:.2f},FN:{:.2f},FP:{:.2f},TN:{:.2f}'.format(tp,fn,fp,tn))
    print('& {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\'.format(accuracy,f1,precision,recall,fpr))
    
if __name__=='__main__':
    sensors_col = ['LIT101', 'LIT301', 'LIT401']  
    label_col = ['Normal/Attack']
    train_path="../../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv"
    test_path="../../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv"
    model_path="./swat_testbed_svm_model_1"
    experiments = ['baseline', 'ntp', 'na']
    costrained_to_future = True
    interval = 1
    
    data = DataPreprocess(train_path, test_path, sensors_col, label_col, interval)
    
    for experiment in experiments:
        print(experiment)
        clf = joblib.load(model_path)
        secml_clf = CClassifierSkLearn(clf)
        metric = CMetricAccuracy()
        #original performance
        y_pred = clf.predict(data.test_data)
        y_pred_secml = secml_clf.predict(data.test_data)
        y_pred_secml[y_pred_secml==0] = -1
        # Evaluate the accuracy of the classifier
        acc = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred))
        acc_secml = metric.performance_score(
            y_true=CArray(data.test_labels), y_pred=CArray(y_pred_secml))

        print("Accuracy on test set original model: {:.2%}".format(acc))
        print("Accuracy on test set secml model: {:.2%}".format(acc_secml))
        
        attack_class = -1 #labels 1 normal, -1 abnormal (ocsvm)
        dmax = 2000  # maximum perturbation
        lb, ub = 100, 1200  # bounds of the attack space. None for unbounded
        # None if untargeted, specify target label otherwise
        y_target = 1 if attack_class == -1 else -1
        
        index_a = np.where(y_pred_secml.tondarray()==-1)
        index_b = np.where(data.test_labels == attack_class)
        if experiment == 'baseline':
            intersection = index_b[0]
        else:
            if experiment == 'ntp':
                intersection = np.intersect1d(index_a, index_b)
            else:
                if experiment == 'na':
                    intersection = index_a[0]
                else:
                    class UnknownExperiment(Exception):
                        pass
                    raise UnknownExperiment('Unknown experiment: allowed strings baseline, ntp, na')

        x=data.test_data[intersection]
        y=data.test_labels[intersection]
        reference = data.train_data[data.train_labels == 1]  
        adv_examples = np.copy(data.test_data)
        elapsed = []
        prev_index = 0
        for x0, y0, index, tq in zip(x,y, intersection, tqdm(range(len(intersection)))):
            if costrained_to_future:
                if prev_index + 1 == index:
                   
                    x0[:3] = adv_examples[prev_index][3:]
            start = time.time()
            x0 = CArray(x0)
            y0 = CArray(y0)
            
            noise_type = 'l1'  # Type of perturbation 'l1' or 'l2'
        
            # Run the evasion attack on x0
            adv_example = x0
            iterations = 0
            prev_sample = CArray([0]*6)
            pred = 0
            
        
            adv_example = x0
            y_pred_secml = secml_clf.predict(adv_example)
            while y_pred_secml == 0:
                scores = secml_clf.forward(adv_example, caching = True)
                grad = secml_clf.backward(scores)
                sign =  np.sign(grad.tondarray())
                speed =  1 if(np.abs(grad)>1e+8).any() else 0.5
                delta = -speed*sign
                adv_example[3:] = adv_example[3:] + CArray(delta[3:])
                iterations = iterations + 1
                prev_sample = adv_example
                y_pred_secml = secml_clf.predict(adv_example)
                
            adv_examples[index] = adv_example.tondarray()
            end = time.time()
            elapsed.append(end - start)
            prev_index = index
        print("Attack finished!")
        elapsed = np.array(elapsed)
        if costrained_to_future: 
            adv_examples.dump('adv_examples_'+experiment+'_costrained.npy')
            y_pred.dump('adv_examples_pred_'+experiment+'_costrained.npy')
            elapsed.dump('elapsed_'+experiment+'_costrained.npy')
        else:
            adv_examples.dump('adv_examples_'+experiment+'.npy')
            y_pred.dump('adv_examples_pred_'+experiment+'.npy')
            elapsed.dump('elapsed_'+experiment+'.npy')
            
        print(experiment)
        print('MEAN: ',np.mean(elapsed), 'STD: ',np.std(elapsed))
        
        y_pred_bef = clf.predict(data.test_data)
        y_pred_secml_bef = secml_clf.predict(data.test_data)
        y_pred_secml_bef[y_pred_secml_bef == 0] = -1
        y_pred = clf.predict(adv_examples)
        y_pred_secml = secml_clf.predict(adv_examples)
        acc_before = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_bef))
        acc_secml_before = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_secml_bef))
        acc = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred))
        acc_secml = metric.performance_score(y_true=CArray(data.test_labels), y_pred=CArray(y_pred_secml))

        print("performance by original model before attack on the attack class:", acc_before)
        print("performance by secml model before attack on the attack class:", acc_secml_before)
        print("performance by original model after attack on the attack class:", acc)
        print("performance by secml model after attack on the attack class:", acc_secml)
        
        one_class_svm_model_testing(data.test_data, data.test_labels, model_path)
        one_class_svm_model_testing(adv_examples, data.test_labels, model_path)