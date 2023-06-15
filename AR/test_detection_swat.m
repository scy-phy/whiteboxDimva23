clear all;
close all;
clc;

Train = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Normal_v1.csv');
Train = Train(14200:end,:);
Test_1 = readtable('../Spoofing Framework/SWAT/SWaT_Dataset_Attack_v1.csv');

%%

Attack_whitebox = readtable('whitebox_attack_all_may22_swat_WBC_baseline.csv');
Attack_whitebox_NTP = readtable('whitebox_attack_all_may22_swat_WBC_NTP.csv');
Attack_whitebox_NA = readtable('whitebox_attack_all_may22_swat_WBC_NA.csv');

Train = downsample(Train, 100);
Test_1 = downsample(Test_1, 100);

Test_1.ATT_FLAG = Attack_whitebox.ATT_FLAG;
order = 80;
gt = 54;
climit = 13.2;
mshift = 2;
column = 20;%3;

Replay = readtable('../Spoofing Framework/SWAT/unconstrained_spoofing/replay.csv');
Replay = downsample(Replay, 100);
Replay.ATT_FLAG = Attack_whitebox.ATT_FLAG;
Random = readtable('../Spoofing Framework/SWAT/unconstrained_spoofing/random_replay.csv');
Random = downsample(Random, 100);
Random.ATT_FLAG = Attack_whitebox.ATT_FLAG;
Stale =  readtable('../Spoofing Framework/SWAT/unconstrained_spoofing/stale.csv');
Stale = downsample(Stale,100);
Stale.ATT_FALG = Attack_whitebox.ATT_FLAG;

%%
disp('Test');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(column, climit, mshift, Train, Test_1, gt, order);
disp('replay');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(column, climit, mshift, Train, Replay, gt, order);
disp('random r');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(column, climit, mshift, Train, Random, gt, order);
disp('stale');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(column, climit, mshift, Train, Stale, gt, order);
disp('whitebox baseline');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(3, climit, mshift, Train, Attack_whitebox, 5, order);
disp('whitebox NTP');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(3, climit, mshift, Train, Attack_whitebox_NTP, 5, order);
disp('whitebox NA');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(3, climit, mshift, Train, Attack_whitebox_NA, 5, order);
%%
writetable(Replay, 'replay_undersample_jan23.csv');
writetable(Random, 'random_replay_undersample_jan23.csv');
writetable(Stale, 'stale_undersample_jan23.csv');