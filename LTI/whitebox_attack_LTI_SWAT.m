%ewquires installing https://github.com/steven2358/sklearn-matlab/tree/master

clear all;

%%Load Train Data
train = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv');
train(:,1)  = [];
ground_truth_train = double(strcmp(table2array(train(:,52)),'Attack'));
train(:,52) = [];


output = train(:,[2,19,29]);
train_lti = train(:,[1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]);
Train = table2array(train_lti);
Output = table2array(output);
Output = Output(20000:end,:);
Train = Train(20000:end,:);
Min_Train=min(Train);
Max_Train=max(Train);
Min_Output=min(Output);
Max_Output=max(Output);

scaler = MinMaxScaler();
scaler.fit(Train);
Norm_train = scaler.transform(Train);
scaler2 = MinMaxScaler();
scaler2.fit(Output);
Norm_output = scaler2.transform(Output);
train_data = iddata(Norm_output,Norm_train,1);
%% Load Test Data

test = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv');
test_orig = test;
test(:,1)  = [];
ground_truth_test = double(strcmp(table2array(test(:,52)),'Attack'));
test(:,52) = [];

test_rolling = test(:,[1,2,6,7,8,9,17,18,19,26,27,28,29,35,36,37,38,39,40,41,42,45,46,47,48]);
Test_rolling = table2array(test_rolling);

output_test = test(:,[2,19,29]);
input_test = test(:,[1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]);
Test = table2array(input_test);
Output_test = table2array(output_test);
Norm_test = scaler.transform(Test);
Norm_output_test = scaler2.transform(Output_test);
test_data = iddata(Norm_output_test,Norm_test,1);

%%
%climits = [10,5];
%mshifts = [6,2];
climits = [10,9,8,6,5];
mshifts = [6,5,4,3,2];
outs = [1,2,3];
%before 13
LTIdetector = LTI_detector(4, climits, mshifts, outs, train_data, ground_truth_train);
%%
LTIpredictions = LTIdetector.anomaly_detection(ground_truth_test, test_data, true);
[accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, double(LTIpredictions));
fprintf('Test Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
fprintf('& %.3f & %.3f & %.3f & %.3f & %.3f', accuracy, f1, precision, recall, fpr)
%%
current_iteration = test_data;
test_new = test_data; 
prev = test_new;
WBC_baseline = false;
WBC_NTP = true;
WBC_NA = false;

indexes = find(ground_truth_test);
elapsed = zeros(length(indexes),1);
[predictions,res] = LTIdetector.anomaly_detection(ground_truth_test, test_data, true);
if WBC_baseline || WBC_NTP
    for index = 1:length(indexes) %
        i = indexes(index);
        if WBC_baseline
            if ground_truth_test(i) == 1 
                [pred, pred_cov]  = predict(LTIdetector.sys, test_new(i-1000:i), 1);
                residual = test_new.y(i,:) - pred.y(end, :);
                p = pred.y(end, :);
                new_e = residual;
                old_e = new_e;
                %avoid leaving the physical plausible feature space
                if  all(p > -1) && all(p<2)
                    tic;
                    test_new.y(i,:) =  test_new.y(i,:) - new_e;
                    time  = toc;
                else
                    tic;
                    positions_high = p > 1;
                    positions_low = p < 0;
                    test_new.y(i, positions_high) = 0;
                    test_new.y(i, positions_low) = 1;
                    time  = toc;
                end
                elapsed(index) = time;
            end
        end
        if WBC_NTP
            if predictions(i) == 1 && ground_truth_test(i) == 1 
                start_index = i-1000;
                if start_index <= 0
                      start_index = 1;
                end
                [pred, pred_cov]  = predict(LTIdetector.sys, test_new(start_index:i), 1);
                res = test_new.y(i,:) - pred.y(end, :);
                new_e = res;
                old_e = new_e+sign(new_e)*0.1;
                p = pred.y(end, :);
                if  all(p > -1) && all(p<2)
                    tic;
                    while max(abs(old_e) > abs(new_e)) && any(abs(res)>0.001)%predictions(i) == 1
                        old_e = new_e;
                        positions = abs(new_e) > 0.001;
                        prev = test_new;
                        step = abs(new_e)/2;
                        test_new.y(i,positions) =  test_new.y(i,positions) - step(positions).*sign(new_e(positions));
                        res(positions) = res(positions)/2;
                        new_e = res;
                    end
                    time  = toc;
                else
                    tic;
                    positions_high = p > 1;
                    positions_low = p < 0;
                    test_new.y(i, positions_high) = 0;
                    test_new.y(i, positions_low) = 1;
                    time  = toc;
                end
                elapsed(index) = time;
                if max(abs(old_e) < abs(new_e))
                   test_new(i) = prev(i);
                end
            end
        end
    end 
else
    indexes = find(predictions);
    for index = 1:length(indexes) %
        i = indexes(index);
        start_index = i-1000;
        if start_index <= 0
              start_index = 1;
        end
        if predictions(i) == 1
                %disp(i);
                start_index = i-1000;
                if start_index <= 0
                      start_index = 1;
                end
                [pred, pred_cov]  = predict(LTIdetector.sys, test_new(start_index:i), 1);
                res = test_new.y(i,:) - pred.y(end, :);
                new_e = res;
                old_e = new_e+sign(new_e)*0.1;
                p = pred.y(end, :);
                if  all(p > -1) && all(p<2)
                    tic;
                    while max(abs(old_e) > abs(new_e)) && any(abs(res)>0.001)%predictions(i) == 1
                        old_e = new_e;
                        positions = abs(new_e) > 0.001;
                        prev = test_new;
                        step = abs(new_e)/2;
                        test_new.y(i,positions) =  test_new.y(i,positions) - step(positions).*sign(new_e(positions));
                        res(positions) = res(positions)/2;
                        new_e = res;
                    end
                    time  = toc;
                else
                    tic;
                    positions_high = p > 1;
                    positions_low = p < 0;
                    test_new.y(i, positions_high) = 0;
                    test_new.y(i, positions_low) = 1;
                    time  = toc;
                end
                elapsed(index) = time;
                
                if max(abs(old_e) < abs(new_e))
                   test_new(i) = prev(i);
                end
        end
    end
end

LTIpredictions = LTIdetector.anomaly_detection(ground_truth_test, test_new, true);
[accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, double(LTIpredictions));
fprintf('Test Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
fprintf('& %.3f & %.3f & %.3f & %.3f & %.3f \n', accuracy, f1, precision, recall, fpr)
%%
if WBC_baseline
    writematrix(elapsed,'elapsed_vector_WBC_baseline_jan23.csv');
    Test_mod = test_orig;
    input_unscale = scaler.inverse_transform(test_new.u);
    output_unscale = scaler2.inverse_transform(test_new.y);
    Test_mod{:,[2,19,29]+1} = output_unscale;
    Test_mod{:, [1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]+1} = input_unscale;
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_baseline.csv');
end
if WBC_NTP
    writematrix(elapsed,'elapsed_vector_WBC_NTP_jan23.csv');
    Test_mod = test_orig;
    input_unscale = scaler.inverse_transform(test_new.u);
    output_unscale = scaler2.inverse_transform(test_new.y);
    Test_mod{:,[2,19,29]+1} = output_unscale;
    Test_mod{:, [1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]+1} = input_unscale;
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_NTP.csv');
end
if WBC_NA
    writematrix(elapsed,'elapsed_vector_WBC_NA_jan23.csv');
    Test_mod = test_orig;
    input_unscale = scaler.inverse_transform(test_new.u);
    output_unscale = scaler2.inverse_transform(test_new.y);
    Test_mod{:,[2,19,29]+1} = output_unscale;
    Test_mod{:, [1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]+1} = input_unscale;
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_NA.csv');
end
%% TEST Evasion attacks from file validation
%% Unconstrained
attack = readtable('../Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv');
attack = readtable('./whitebox_attack_all_may22_swat_WBC_NA.csv');
attack(:,1)  = [];
attack(:,52) = [];

output_attack_LTI = attack(:,[2,19,29]);
attack = attack(:,[1,6,7,8,9,17,18,26,27,28,35,36,37,38,39,40,41,42,45,46,47,48]);
Attack = table2array(attack);
Output_attack = table2array(output_attack_LTI);

Norm_attack = scaler.transform(Attack);
Norm_output_attack = scaler2.transform(Output_attack);
attack_data = iddata(Norm_output_attack,Norm_attack,1);

LTIpredictions_attack = LTIdetector.anomaly_detection(ground_truth_test, attack_data, true);

[accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, double(LTIpredictions_attack));
%area(overall_attack);
fprintf('Evasion Attack Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
fprintf('& %.3f & %.3f & %.3f & %.3f & %.3f', accuracy, f1, precision, recall, fpr)
%%
figure;
plot(ground_truth_test, 'LineWidth',4);
hold on;
area(overall_attack);

%%
unscale = @(data, l, u, inmin, inmax) (data - l) * (inmax - inmin) / (u - l) + inmin;

%%
if ground_truth_test(i) == 1 
        if LTIpredictions(i) == 1
            count_with_i = 0;
            %disp(i);
            [e,r] = resid(test_new(1:i),LTIdetector.sys); 
            new_e = e.y(end,:);
            old_e = new_e;
            while (LTIpredictions(i) == 1)% && max(abs(old_e) >= abs(new_e))% && abs(test_new(i) - test(i)) <= 10)
                old_e = new_e;
                count_with_i = count_with_i + 1;
                [pred, pred_cov]  = predict(LTIdetector.sys, test_new(1:i), 1);
                res = test_new.y(i,:) - pred.y(end, :);
                positions = abs(res) > 0.01;
                test_new.y(i,positions) =  test_new.y(i,positions) - 0.5*res(positions);
                [e,r] = resid(test_new(1:i),LTIdetector.sys); 
                new_e = e.y(end,:);
                new_LTIpredictions =  LTIdetector.anomaly_detection(ground_truth_test(1:i), test_new(1:i), false);
                LTIpredictions(i)=new_LTIpredictions(end);
            end
        end
    end