% -------------------------------------------------------------------------
% calculateFeatures.m
% -------------------------------------------------------------------------
% DESCRIPTION:
% This script makes the feature calculation for an audio folder recordered
% using ASICA app.
%
% The full raw audio data recorded is not available because it contains 
% audios from underage patients. An example of a female woman, age 31, its
% provided as demo.
% 
% The features used in this work are stored as .mat files:
% features_36P_36H.mat
% features_36P_36H9C.mat
% features_36P_49H.mat
% features_36P_49H9C.mat
% -------------------------------------------------------------------------
% INPUTS:
% -------------------------------------------------------------------------
% OUTPUTS:
% - dataBaseWorkspace.mat: .mat file with:
%       - classDataTable: table with patient data, class, age and fs.
%       - databaseTable: tabla with patient features.
% -------------------------------------------------------------------------
close all; clear; clc;

addpath(genpath('lib'));
addpath(genpath('audio'));

audioPath = fullfile(pwd,'audio');


% Generate a struct with patients audios
patientName = 'fem39000';
patientStc = createNewPatient(patientName);


% Data arrays used to store features 
featuresGlobal = [];        
featuresNamesGlobal = [];	

% Feature calculation

strDisp = sprintf(['\t Calculating audio: ' patientName]);
disp(strDisp);

% Patient data
age = string(patientStc.info.age);
class = string(patientStc.info.class);
fs = patientStc.info.fs;
codePatient = string([patientStc.info.class patientStc.info.age patientStc.info.id]); 
% codePatient = patientName;
        
sNames = fieldnames(patientStc.audio);

tic
for m = 1:length(sNames)   
    
    features = [];             
    featuresNames = [];           
    audio = patientStc.audio.(sNames{m});
    
   
    % Proprocessing audio to delete silence parts
    [beg1, end1, beg2, end2, ~, ~] = silenceDetectorUtterance(audio,fs, 0.025, 0.015);
    if end2-beg2 > 1    % if audio last less than 1s use full length data.
        audio = audio(floor(fs*beg2):ceil(fs*end2));
    end % if 
   
    
    if contains(sNames{m},'T1_00_Conteo')
        [features, featuresNames] = computeFeatures(audio,fs,'_T1_10');
    elseif contains(sNames{m},'T2')  
        [features, featuresNames] = computeFeatures(audio,fs,...
            ['_T2_' sNames{m}(end-1:end)]);
    elseif contains(sNames{m},'T3')     
        if contains(sNames{m},'Modelo_f') 
            letterAux = 'f';
        elseif contains(sNames{m},'Modelo_s') 
            letterAux = 's';
        end % if 
        [features, featuresNames] = computeFeatures(audio,fs,['_T3' letterAux]);
    elseif contains(sNames{m},'T4')     
        [features, featuresNames] = computeFeatures(audio,fs,'_T4');
    elseif contains(sNames{m},'T5') && ...
            str2double(extractBetween(sNames{m},'T5_','_PalModelo'))<=20 % palabras
        [features, featuresNames] = ...
            computeFeatures(audio,fs,['_T5_' extractAfter(sNames{m},"Modelo_")]); 
    elseif contains(sNames{m},'T6') && ...
            str2double(extractBetween(sNames{m},'T6_','_Frases'))<=18 % frases
        [features, featuresNames] = ...
            computeFeatures(audio,fs,['_T6_' extractAfter(sNames{m},"Modelo_")]); 
    end % if contains 
    
    % Acumulate feature generated
    featuresGlobal = [featuresGlobal features];
    featuresNamesGlobal = [featuresNamesGlobal featuresNames];
end % for m
toc


% Save data generated as matlab tables
databaseTable = array2table(featuresGlobal,'RowNames',codePatient,'VariableNames',featuresNamesGlobal');
classDataTable = array2table([class age fs],'RowNames',codePatient,'VariableNames',["class" "age" "fs"]);

databaseTable = sortrows(databaseTable,'RowNames');
classDataTable = sortrows(classDataTable,'RowNames');

% Save Workspace
save(fullfile(pwd,'dataBaseWorkspace.mat'),'databaseTable','classDataTable');




