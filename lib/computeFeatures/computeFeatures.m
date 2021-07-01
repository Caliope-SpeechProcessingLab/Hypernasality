function [features, featuresNames] = computeFeatures(audio,fs,str)
% -------------------------------------------------------------------------
% function [features, featuresNames] = computeFeatures(audio,fs,str)
% -------------------------------------------------------------------------
% DESCRIPTION:
% This functions computes the features of audio data.
% -------------------------------------------------------------------------
% INPUTS:
% - audio: audio stored
% - fs: sample frequency of audio
% - str: string name used to identify differents features
% -------------------------------------------------------------------------
% OUTPUTS: 
% - features: features calculated
% - featuresNames: names of features calculated
% -------------------------------------------------------------------------

%VLHR
vlhr400 = calculateVLHR(audio,fs,'cutoff',400);
vlhr500 = calculateVLHR(audio,fs,'cutoff',500);
vlhr600 = calculateVLHR(audio,fs,'cutoff',600);
vlhr700 = calculateVLHR(audio,fs,'cutoff',700);
vlhr800 = calculateVLHR(audio,fs,'cutoff',800);
vlhr900 = calculateVLHR(audio,fs,'cutoff',900);


% MFCC
[coeffs,~,~,~] = mfcc(audio,fs,'LogEnergy','Replace',...
    'Window',hamming(round(fs*0.025),'periodic'),'OverlapLength',round(fs*0.015),'NumCoeffs',13);
% 25 ms windows, 15 ms windows overlapping
mfcc_values = mean(coeffs,1);


% Formants frequency and bandwidth 
[formants, bwFormants] = calculateFormants(audio,fs);
distFromt = abs([(formants(1)-formants(2)) (formants(1)-formants(3)) (formants(2)-formants(3))]);

    
% Features calculated
features = [mfcc_values vlhr400 vlhr500 vlhr600 vlhr700 vlhr800 vlhr900...
    formants bwFormants distFromt];

featuresNames = ["mfcc1","mfcc2","mfcc3","mfcc4","mfcc5","mfcc6","mfcc7",...
    "mfcc8","mfcc9","mfcc10","mfcc11","mfcc12","mfcc13",...
    "vlhr400","vlhr500","vlhr600","vlhr700","vlhr800","vlhr900",...
    "f1","f2","f3","bwf1","bwf2","bwf3",...
    "f1-f2","f1-f3","f2-f3"];

featuresNames = strcat(featuresNames,str);

end % function



























