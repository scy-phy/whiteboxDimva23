clear all;
close all;
clc;

Train = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv');
Train = Train(14200:end,:);
Test_1 = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv');

Train = downsample(Train, 100);
Test_1 = downsample(Test_1, 100);

order = 80;
gt = 54;
climit = 13.2;
mshift = 2;
column = 20;

train = Train(:, column);
train = table2array(train);
train_idd = iddata(train, [], 1);
sys = ar(train_idd, order, 'ls');
test = Test_1(:, column);
test = table2array(test);
[mfcn, sfnc] = cusum_detector(train, sys);
%%
test_new = test; 
current_iteration = test;
ground_truth_test = table2array(Test_1(:, gt));

WBC_baseline = true;
WBC_NTP = false;
WBC_NA = false;

resids = [];
elapsed = [];
for i = 1:length(test_new)
   start_index = i-order;
   if start_index <= 0
       start_index = 1;
   end
   [prediction,resids] =  test_detection_with_old_resids(mfcn, sfnc, test_new(start_index:i), sys, climit, mshift, resids);
   if WBC_baseline
       if ground_truth_test(i) == 1 
           disp(i);
           old_e = abs(resids(i))+0.1;
           y1 = abs(resids(i));
           grad = resids(i); 
           tic;
           test_new(i) =  test_new(i) - grad;
           time  = toc;
           elapsed = [elapsed, time];
           resids(i)=[];
           [prediction, resids] =  test_detection_with_old_resids(mfcn, sfnc, test_new(start_index:i), sys, climit, mshift, resids);
           disp(prediction(i));
       end
   end
   
   if WBC_NTP
       if prediction(i) == 1 && ground_truth_test(i) == 1 
           disp(i);
           old_e = abs(resids(i))+0.1;
           y1 = abs(resids(i));
           grad = resids(i); %test_new(i)-pred(end); %test new is the value that we can change x_hat
           tic;
           while abs(old_e) > abs(y1) && prediction(i) == 1
                test_new(i) = current_iteration(i);
                old_e = resids(i);
                step = abs(old_e)/2;
                if sqrt(old_e*old_e) <= 5
                    step = 1;
                end
                current_iteration(i) =  current_iteration(i) - step*sign(grad);
                resids(i)=[];
                [prediction, resids] =  test_detection_with_old_resids(mfcn, sfnc, current_iteration(start_index:i), sys, climit, mshift, resids);
                y1=resids(i);
           end
           time  = toc;
           elapsed = [elapsed, time];
           %if we exit because prediction == 0 then set the latest sample as
           %adversarial
           if prediction(i)==0
               test_new(i) = current_iteration(i);
           end
           resids(i)=[];
           [prediction, resids] =  test_detection_with_old_resids(mfcn, sfnc, test_new(start_index:i), sys, climit, mshift, resids);
           disp(prediction(i));
       end
   end
   if WBC_NA
       if prediction(i) == 1 %WBC no label cheap
           disp(i);
           old_e = abs(resids(i))+0.1;
           y1 = abs(resids(i));
           grad = resids(i); %test_new(i)-pred(end); %test new is the value that we can change x_hat
           tic;
           while abs(old_e) > abs(y1) && prediction(i) == 1
                test_new(i) = current_iteration(i);
                old_e = resids(i);
                step = abs(old_e)/2;
                if sqrt(old_e*old_e) <= 5
                    step = 1;
                end
                current_iteration(i) =  current_iteration(i) - step*sign(grad);
                resids(i)=[];
                [prediction, resids] =  test_detection_with_old_resids(mfcn, sfnc, current_iteration(start_index:i), sys, climit, mshift, resids);
                y1=resids(i);
           end
           time  = toc;
           elapsed = [elapsed, time];
           %if we exit because prediction == 0 then set the latest sample as
           %adversarial
           if prediction(i)==0
               test_new(i) = current_iteration(i);
           end
           resids(i)=[];
           [prediction, resids] =  test_detection_with_old_resids(mfcn, sfnc, test_new(start_index:i), sys, climit, mshift, resids);
           disp(prediction(i));
       end
   end
end
if WBC_baseline
    writematrix(elapsed,'elapsed_vector_WBC_baseline.csv');
    Test_mod = Test_1;
    Test_mod(:,column) = array2table(test_new);
    writetable(Test_mod,'whitebox_attack_all_may22_swat_WBC_baseline.csv');
end
if WBC_NTP
    writematrix(elapsed,'elapsed_vector_WBC_NTP.csv');
    Test_mod = Test_1;
    Test_mod(:,column) = array2table(test_new);
    writetable(Test_mod,'whitebox_attack_all_may22_swat_WBC_NTP.csv');
end
if WBC_NA
    writematrix(elapsed,'elapsed_vector_WBC_NA.csv');
    Test_mod = Test_1;
    Test_mod(:,column) = array2table(test_new);
    writetable(Test_mod,'whitebox_attack_all_may22_swat_WBC_NA.csv');
end
%%
disp(max(test-test_new));
disp(std(test-test_new));
plot(test, 'color','r');
hold on;
plot(test_new, 'color', 'b');
hold off;

%%
figure;
new = predict(sys, test_new); 
plot(new);
hold on;
plot(test_new, 'color', 'r');
hold off;
figure;
old = predict(sys, test); 
hold on;
plot(old);
plot(test, 'color', 'b');
hold off;

%%
function [mfcn, sfnc] = cusum_detector(train, sys)
    [e_train,r_train] = resid(train, sys);
    mfcn = mean(e_train.y);
    sfnc = std(e_train.y);
end
%%
function [prediction_test, e_test]=test_detection(mfnc, sfnc, test, sys, climit, mshift)
    [e_test,r_test] = resid(test, sys);
    [iupper_test, ilower_test] = cusum(e_test.y,climit,mshift,mfnc,sfnc, 'all');
    prediction_test = merge_cusum_results(test, iupper_test, ilower_test);
end

function [prediction_test,resids]=test_detection_with_old_resids(mfnc, sfnc, test, sys, climit, mshift, resids)
    [e_test,r_test] = resid(test, sys);
    resids(end+1) = e_test.y(end);
    [iupper_test, ilower_test] = cusum(resids',climit,mshift,mfnc,sfnc, 'all');
    prediction_test = merge_cusum_results(resids, iupper_test, ilower_test);
end
function prediction = merge_cusum_results(ground_truth, iupper, ilower)
        prediction = zeros([length(ground_truth) 1]);
        for i = 1:length(ground_truth)
            prediction(i) = ismember(i, iupper);
            if prediction(i)==0
                prediction(i) = ismember(i, ilower);
            end
        end
end
