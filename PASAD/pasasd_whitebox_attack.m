%the detection code is based on https://github.com/mikeliturbe/pasad
%whitebox attack

Train = readtable('../data/SWAT/SWAT_train.csv');
Test = readtable('../data/SWAT/SWAT_test.csv');
train = table2array(Train(:, 3));
test = table2array(Test(:,3));
ground_truth = table2array(Test(:, 5));
params = [30000, 5000];
statistical_dim = 10;

%% Training Phase
% Obtaining a mathematical representation of the process behavior by
% determining the statistical dimension and the principal components of the
% signal subspace.

s = train;
testing = test;
ground_truth = ground_truth;

I = params;
N = I(1); L = I(2);
T = length(testing);
K = N-L+1;

% Defining custom colors
bk = [.24 .24 .36]; 
rd = [1 .5 .5]; 
gr = [.8 1 .1]; 
bl = [.7 .85 1]; 

% Range vector corresponding to the sensor measurements affected by the
% attack (Should be adapted according to the entered time series. Defaulted
% to TE non-hf series).
atck_rg = find(ground_truth == 1);

% Constructing the (Hankel) trajectory matrix and solving its Singular
% Value Decomposition (SVD).
X = hankel(s(1:L),s(L:N));
disp('SVD decomposition started ...');tic
[t,e,~] = svd(X); 
ev = diag(e);
disp('SVD decomposition complete');toc

% Determining the statistical dimension of the time series.
es = (ev(2:end)./sum(ev(2:end)))*100;
figure
plot(es,'color',[.4 .4 .4],'linewidth',2),hold on,
plot(es,'rx','color',[1 .4 .2]);
xlabel('Number of eigenvalues');
ylabel('Eigenvalue share')
title('Scree plot');
set(gca,'fontsize',16);
r = statistical_dim;%input('Specify the statistical dimension: ');
close gcf
disp('Training PASAD is complete.');

% Constructing the matrix whose columns form an orthonormal basis for the
% signal subspace.
U = t(:,(1:r));

% Computing the centroid of the cluster formed by the training lagged
% vectors in the signal subspace.
c = mean(X,2);
utc = U'*c;

% A vector containing the normalization weights for computing the squared
% weighted Euclidean distance in the detection phase.
nev = sqrt(ev(1:r)./sum(ev(1:r)));

% Reconstring the approximate signal using the diagonal averaging step in
% Singular Spectrum Analysis (SSA).
disp('Reconstructing signal ...');tic
ss = U*(U'*X);

sig = zeros(N,1);  

for k = 0:L-2
    for m = 1:k+1
        sig(k+1) = sig(k+1)+(1/(k+1))*ss(m,k-m+2);
    end
end

for k = L-1:K-1
    for m = 1:L
        sig(k+1) = sig(k+1)+(1/(L))*ss(m,k-m+2);
    end
end

for k = K:N
    for m = k-K+2:N-K+1
        sig(k+1) = sig(k+1)+(1/(N-k))*ss(m,k-m+2);
    end
end
disp('Signal reconstruction complete');toc

%% Detection Phase
% Tracking the distance from the centroid by iteratively computing the
% departure score for every test vector.

disp('Testing started...');
d = zeros(T,1);

% Constructing the first test vector.
x = testing(N-L+1:N);
testing_adv = testing;
prev_i = 0;
count = 0;
elapsed = [];
WBC_baseline = true;
WBC_NTP = false;
WBC_NA = false;
for i = 1:T
    % Constructing the current test vector by shifting the elements to
    % the left and appending the current sensor value to the end.
    x = x([2:end 1]);
    x(L) = testing(i);%s(i); 
    % Computing the difference vector between the centroid of the
    % cluster and the projected version of the current test vector.
    y = utc - U'*x;
    % Computing the weighted norm of the difference vector.
    y = nev.*y;
    distance = y'*y;
    %d_original(i) = distance;
    d(i) = distance;
    new_d = distance;
    old_d = distance+0.1;
    minimum = min(train)-10;
    maximum = max(train)+10;
    last_exceeded = -1;
    if WBC_baseline
        if ground_truth(i) == 1
            f = x(L)-c(L); 
            min_d = old_d;
            min_x = x(L);
            if x(L)<minimum
                x(L) = minimum;
            end
            if x(L) > maximum
                x(L) = maximum;
            end
            x_ad = x;
            tic;
            while(old_d > new_d && x_ad(L)>=minimum && x_ad(L)<=maximum)
                old_d = new_d;
                x(L) = x_ad(L); %save the last good adversarial example
                x_ad(L) = (x_ad(L) - 0.5*sign(f));
                y = U'*(c - x_ad);
                y = nev.*y;
                new_d = y'*y;
            end
            time  = toc;
            elapsed = [elapsed, time];
            d(i) = old_d;  
            testing_adv(i) = x(L);
        end
    end
    
    if WBC_NTP
        if (d(i)>=2800000 && ground_truth(i) == 1)
            f = x(L)-c(L); 
            min_d = old_d;
            min_x = x(L);
            if x(L)<minimum
                x(L) = minimum;
            end
            if x(L) > maximum
                x(L) = maximum;
            end
            x_ad = x;
            tic;
            while(old_d > new_d && new_d>=2800000)
                old_d = new_d;
                x(L) = x_ad(L);
                x_ad(L) = (x_ad(L) - 0.5*sign(f));
                y = U'*(c - x_ad);
                y = nev.*y;
                new_d = y'*y;
            end
            time  = toc;
            elapsed = [elapsed, time];
            if old_d < new_d
                d(i)= old_d;
            end
            if new_d<2800000
               d(i) = new_d;  
               x(L) = x_ad(L);
            end
            testing_adv(i) = x(L);
        end
    end
    if WBC_NA
        if d(i)>=2800000 
            f = x(L)-c(L);
            min_d = old_d;
            min_x = x(L);
            if x(L)<minimum
                x(L) = minimum;
            end
            if x(L) > maximum
                x(L) = maximum;
            end
            x_ad = x;
            tic;
            while(old_d > new_d && new_d>=2800000)
                old_d = new_d;
                x(L) = x_ad(L);
                x_ad(L) = (x_ad(L) - 0.5*sign(f));
                y = U'*(c - x_ad);
                y = nev.*y;
                new_d = y'*y;
            end
            time  = toc;
            elapsed = [elapsed, time];
            d(i) = new_d;  
            if old_d < new_d
                d(i)= old_d;
            end
            if new_d<2800000
               x(L) = x_ad(L);
            end
            testing_adv(i) = x(L);
        end
    end
end

if WBC_baseline
    writematrix(elapsed,'elapsed_vector_WBC_baseline.csv');   
    Test_mod = Test;
    Test_mod(:,3) = array2table(testing_adv);
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_baseline.csv') 
end

if WBC_NTP
    writematrix(elapsed,'elapsed_vector_WBC_NTP.csv');   
    Test_mod = Test;
    Test_mod(:,3) = array2table(testing_adv);
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_NTP.csv') 
end

if WBC_NA
    writematrix(elapsed,'elapsed_vector_WBC_NA.csv');   
    Test_mod = Test;
    Test_mod(:,3) = array2table(testing_adv);
    writetable(Test_mod,'whitebox_attack_all_jan23_swat_WBC_NA.csv') 
end
disp('Testing complete.');%toc
%% Plof of the result
indexes = atck_rg(diff(atck_rg) > 1);
intervals = zeros([length(indexes)+1 2]);
intervals(1, :) =  [1 find(atck_rg == indexes(1))];
for i = 2:length(indexes)
    intervals(i,:) = [find(atck_rg==indexes(i-1))+1 find(atck_rg == indexes(i))];
end
intervals(end, :) = [find(atck_rg == indexes(end))+1 length(atck_rg)];
s = testing_adv;
figure
ax1 = subplot(2,1,1);hold on
plot(Test.DATETIME, s,'color',bk);
ylim([600 1200]);
for dim = 1:size(intervals, 1)
    plot(Test.DATETIME(atck_rg(intervals(dim, 1) : intervals(dim,2))), s(atck_rg(intervals(dim, 1): intervals(dim,2))), 'color', rd);
end

ylabel('Sensor Meas.');
set(gca,'fontsize',26,'linewidth',1.5);
legend('Normal','Attack');
ax2 = subplot(2,1,2);hold on 
trans = plot(Test.DATETIME, d_original,'color', [0.75 0.75 0.75],'linewidth',2);
alpha(trans, .5);

plot(Test.DATETIME, d,'color','b','linewidth',2);

plot(Test.DATETIME, ones(size(Test))*3000000, '--r');
ylim([0 10000000]);
xlabel('Observation Index');
ylabel('Departure Score');
legend('Original', 'Attack', 'Threshold');
set(gca,'fontsize',26,'linewidth',1.5);

linkaxes([ax1,ax2],'x');

detection_indexes = (d>=3000000);
conf_matrix = confusionmat(ground_truth, double(detection_indexes), 'Order', [1,0]);
hold off;

tp = conf_matrix(1, 1);
tn = conf_matrix(2, 2);
fp = conf_matrix(2, 1);
fn = conf_matrix(1, 2);

accuracy = (tp+tn)/(tp+tn+fp+fn);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1score = 2*((precision*recall)/(precision+recall));
fpr = fp/(fp+tn);
fprintf('Accuracy: %.3f F1-score: %.3f Precision: %.3f Recall: %.3f FPR: %.3f\n', accuracy, f1score, precision, recall, fpr);
