%This detector code is based on https://github.com/scy-phy/ICS_Generic_Concealment_Attacks/blob/main/Countermeasure/LTI_detector.m
classdef LTI_detector
    properties
        sys
        climits
        mshifts
        mfnc
        sfnc
    end
    methods
        function detector = LTI_detector(order, cl, ms, out, data, ground_truth_train)
            % Subspace Sytem Identification Model
            nx = order;
            detector.sys = n4sid(data,nx);

            % Compute residuals of 1-step ahead prediction
            [e_train,~] = resid(data, detector.sys);
            detector.mfnc = mean(e_train.y);
            detector.sfnc = std(e_train.y);
            % Search for best Climit and mshift
            climits_iter= cl; 
            mshifts_iter=ms;
            outs=out;
            best = repmat([0,0,1000], length(outs),1);
            for climit = climits_iter
                for mshift=mshifts_iter
                    for out=outs
                        [iupper_train, ilower_train] = cusum(e_train.y(:,out), climit, ...
                            mshift, detector.mfnc(out), detector.sfnc(out), 'all');
                        prediction_train = merge_cusum_results(ground_truth_train, ...
                            iupper_train, ilower_train);
                        [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores...
                            (ground_truth_train, prediction_train);
                        if best(out,3) >= fpr
                            best(out,1)=climit;
                            best(out,2)=mshift;
                            best(out,3)=fpr; 
                        end
                    end
                end
            end
            detector.climits=best(:,1);
            detector.mshifts=best(:,2);
        end
        
        function detector = find_best(detector, cl, ms, out, data, ground_truth_train)
            
            [e_train,~] = resid(data, detector.sys);
            detector.mfnc = mean(e_train.y);
            detector.sfnc = std(e_train.y);
            % Search for best Climit and mshift
            climits_iter= cl; 
            mshifts_iter=ms;
            outs=out;
            best = repmat([0,0,1000], length(outs),1);
            for climit = climits_iter
                for mshift=mshifts_iter
                    for out=outs
                        [iupper_train, ilower_train] = cusum(e_train.y(:,out), climit, ...
                            mshift, detector.mfnc(out), detector.sfnc(out), 'all');
                        prediction_train = merge_cusum_results(ground_truth_train, ...
                            iupper_train, ilower_train);
                        [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores...
                            (ground_truth_train, prediction_train);
                        if best(out,3) >= fpr
                            best(out,1)=climit;
                            best(out,2)=mshift;
                            best(out,3)=fpr; 
                        end
                    end
                end
            end
            detector.climits=best(:,1);
            detector.mshifts=best(:,2);
        end
    
        function [predictions,residuals] = anomaly_detection(detector, ground_truth, data, print)
            predictions = zeros([length(ground_truth) 1]);
            outs = 1:length(detector.climits);
            max_uppersum = zeros([length(data) 1]);
            min_lowersum = zeros([length(data) 1]);
            [e_test,~] = resid(data, detector.sys);
            for out=outs
                [iu_test, il_test] = cusum(e_test.y(:,out), detector.climits(out), detector.mshifts(out), detector.mfnc(out), detector.sfnc(out), 'all');
                [iupper_test, ilower_test, UPPERSUM, LOWERSUM] = bounded_cusum(e_test.y(:,out), detector.climits(out), detector.mshifts(out), detector.mfnc(out), detector.sfnc(out), 'all');
                prediction = merge_cusum_results(ground_truth, iupper_test, ilower_test);
                predictions = predictions | prediction; 
                rescale_upper_limit = (detector.climits(out)+2)/detector.climits(out);
                Norm_UPPERSUM = rescale(UPPERSUM, 0, rescale_upper_limit);
                Norm_LOWERSUM = rescale(LOWERSUM, -rescale_upper_limit,0);
                max_uppersum = max(Norm_UPPERSUM, max_uppersum);
                min_lowersum = min(Norm_LOWERSUM, min_lowersum);
            end
            residuals = e_test.y;
            if print == true
                plotchart(max_uppersum, min_lowersum, detector.mfnc(out), detector.sfnc(out), 1 ,iupper_test,ilower_test);
                [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth, double(predictions));
                fprintf('Overall Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
            end
        end
        
    end
end

%class related functions
function prediction = merge_cusum_results(ground_truth, iupper, ilower)
        prediction = zeros([length(ground_truth) 1]);
        for i = 1:length(ground_truth)
            prediction(i) = ismember(i, iupper);
            if prediction(i)==0
                prediction(i) = ismember(i, ilower);
            end
        end
end

function  plotchart(uppersum,lowersum,tmean,tdev,climit,iupper,ilower)
    newplot
    n = length(uppersum);
    hLines = plot(1:numel(uppersum),uppersum(:),1:numel(lowersum),lowersum(:));
    line([1 n],climit*[1 1],'LineStyle',':','Color',hLines(1).Color);
    line([1 n],-climit*[1 1],'LineStyle',':','Color',hLines(2).Color);

    if ~isempty(iupper)
      line(iupper,uppersum(iupper),'LineStyle','none');
    end

    if ~isempty(ilower)
      line(ilower,lowersum(ilower), 'LineStyle','none');
    end

    hAxes = hLines(1).Parent;
    hAxes.YLim = [min(-climit-1,hAxes.YLim(1)) max(climit+1,hAxes.YLim(2))];

end

