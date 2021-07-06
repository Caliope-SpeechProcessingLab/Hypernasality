% -------------------------------------------------------------------------
% clasificationHypernasality.m
% -------------------------------------------------------------------------
% DESCRIPTION:
% This script makes the hypernasality clasification with RF classifier.
%
% The features used in this work are stored in features_39P_39H.mat
% -------------------------------------------------------------------------
% INPUTS:
% -------------------------------------------------------------------------
% OUTPUTS: Classification results are shown in command window.
% -------------------------------------------------------------------------
close all; clear; clc;

addpath(genpath('lib'));


% Comment the lines to select features data selected
tableGlobal = load('features_39P_39H');

sNames = fieldnames(tableGlobal);
tableGlobal = tableGlobal.(sNames{1});


% Define the audios we test individually
testToDo = [];
subTest = {};

sNamesVariables = tableGlobal.Properties.VariableNames(2:end);
sNamesVariablesShort = unique(cellfun(@(x) extractAfter(x(1:end),'_'),...
    sNamesVariables,'UniformOutput',false));
testToDo = string(sNamesVariablesShort);
subTest = sNamesVariablesShort;

% Table 2. Optimal utterance lists
base1 = "T2_ka";
base2 = "T5_dedo";

testToDo = [testToDo,"Table2_T2_ka","Table2_T2_ka_dedo","Table2_add_T2_ki","Table2_add_T2_ta",...
    "Table2_add_T2_ti","Table2_add_T6_ADavidLeDuele","Table2_add_T3f","Table2_add_T6_LaliyLunaLeen",...
    "Table2_add_T5_pie","Table2_add_T6_SusiSaleSola",...
    "Table2_add_T6_SiLlueveleLlevo","Table2_add_T5_pez"];

subTest{end+1} = [base1];
subTest{end+1} = [base1,base2];
subTest{end+1} = [base1,base2,"T2_ki"];
subTest{end+1} = [base1,base2,"T2_ta"];
subTest{end+1} = [base1,base2,"T2_ti"];
subTest{end+1} = [base1,base2,"T6_ADavidLeDuele"];
subTest{end+1} = [base1,base2,"T3f"];
subTest{end+1} = [base1,base2,"T6_LaliyLunaLeen"];
subTest{end+1} = [base1,base2,"T5_pie"];
subTest{end+1} = [base1,base2,"T6_SusiSaleSola"];
subTest{end+1} = [base1,base2,"T6_SiLlueveleLlevo"];
subTest{end+1} = [base1,base2,"T5_pez"];

% Table 3. Utterances and classifiers used in the HN Score (7 + Sel)
testToDo = [testToDo,"Table3_T2_ki","Table3_T2_ka","Table3_T2_ta"];

subTest{end+1} = ["T2_ki"];
subTest{end+1} = ["T2_ka"];
subTest{end+1} = ["T2_ta"];


% Variables initialization
classNasal = logical(tableGlobal.classNasal);
tableGlobal = tableGlobal(:,2:end);

featuresStc = struct();
featuresNamesStc = struct();

for k = 1:length(testToDo)
    indexVariableNames = contains(tableGlobal.Properties.VariableNames,subTest{k});
    
    featuresStc.(testToDo(k)) = table2array(tableGlobal(:,indexVariableNames));
    featuresNamesStc.(testToDo(k)) = tableGlobal.Properties.VariableNames(:,indexVariableNames);
end % for k
patientNames = tableGlobal.Properties.RowNames;



% Clasification process
kfaux = tabulate(double(classNasal)); tabulate(double(classNasal))
k_fold = 5;
rng('default'); cPartition = cvpartition(classNasal,'KFold',k_fold)

numFeaturesRedux = 20;
featuresNamesSelectedStc = struct();

confMatrixRFStc = struct();
sNamesErroresRFStc = struct();

for indexTest = 1:length(testToDo)
    tic
    % Cada prueba se hace por separado
    disp(['Performing test ' num2str(indexTest) ': '  char(testToDo(indexTest))]);
    
    sNamesErroresRFStc.(testToDo(indexTest)) = [];
    featuresNamesSelectedStc.(testToDo(indexTest)) = [];
    
    for k = 1:k_fold
        disp(['Doing k_fold: ' num2str(k)]);
        
        idx_train = training(cPartition,k);
        idx_test = test(cPartition,k);
        
        trainData = featuresStc.(testToDo(indexTest))(idx_train,:);
        classTrain = classNasal(idx_train,:);
        sNamesTrain = patientNames(idx_train);
        testData = featuresStc.(testToDo(indexTest))(idx_test,:);
        classTest = classNasal(idx_test,:);
        sNamesTest = patientNames(idx_test);
        
        % Features selection
        featuresNames = featuresNamesStc.(testToDo(indexTest));
        % 'ttest' — Absolute value two-sample t-test with pooled variance estimate.
        [idx, absoluteValueCriteria] = rankfeatures(trainData', classTrain',...
            'Criterion','ttest','CrossNorm','meanvar');
        
        trainData = trainData(:,idx(1:numFeaturesRedux));
        testData = testData(:,idx(1:numFeaturesRedux));
        
        featuresNamesSelectedStc.(testToDo(indexTest)) = [featuresNamesSelectedStc.(testToDo(indexTest)) ...
            featuresNames(idx(1:numFeaturesRedux))];
        
        % Normalización
        mediaTrain = mean(trainData,'omitnan'); stdTrain = std(trainData,0,'omitnan');
        trainDataNorm = (trainData-mediaTrain)./stdTrain;
        testDataNorm = (testData-mediaTrain)./stdTrain;
        
        % PCA
        trainDataRedux = trainDataNorm;
        testDataRedux = testDataNorm;
        
        [wcoeff,score,latent,tsquared,explained,mu] = pca(trainDataRedux,'Algorithm','svd');
        
        explainedSum = cumsum(explained);
        paretoPCA(k) = find(explainedSum > 95 ,1);
        
        trainDataPCA = trainDataRedux * wcoeff(:,1:paretoPCA(k));
        testDataPCA = testDataRedux * wcoeff(:,1:paretoPCA(k));
        
        % Classifiers
        randTrain = randperm(size(trainDataPCA,1));
        randTest = randperm(size(testDataPCA,1));
        
        % RF
        dataRFTrain = trainDataPCA(randTrain,:);
        classDataRFTrain = classTrain(randTrain);
        sNamesTrainRF = sNamesTrain(randTrain);
        dataRFTest = testDataPCA(randTest,:);
        classDataRFTest = classTest(randTest);
        sNamesTestRF = sNamesTest(randTest);
        
        numRandomForest = 500;
        RFModel = TreeBagger(numRandomForest,dataRFTrain,classDataRFTrain,...
            'Cost',[0 1; 1 0],...
            'Method','classification','OOBPrediction','on','Prior','Empirical');
        
        [labelClasificador,~] = predict(RFModel,dataRFTest);
        labelClasificador = str2double(labelClasificador);
        
        confusM = confusionmat(double(classDataRFTest),labelClasificador);
        sNamesErrorRF = sNamesTestRF(double(classDataRFTest)~=labelClasificador);
        
        confMatrixRFStc.(testToDo(indexTest))(:,:,k) = confusM;
        sNamesErroresRFStc.(testToDo(indexTest)) = ...
            [sNamesErroresRFStc.(testToDo(indexTest)); sNamesErrorRF];
        
    end % for k
    
    toc
end % for indexTest
save(fullfile(pwd,'resultadosArtículo'));


%% Creamos una tabla con todos los usuarios y las pruebas que han pasado
tablaErroresRF = zeros(length(patientNames),length(testToDo));

for indexErrTest = 1:length(testToDo)  
    errorsAux = sNamesErroresRFStc.(testToDo(indexErrTest));
    indexErr = contains(patientNames,errorsAux);
    
    tablaErroresRF(indexErr,indexErrTest) = 1;
end % for k

errorsTableRF = array2table(tablaErroresRF,'RowNames',patientNames,'VariableNames',testToDo');
writetable(errorsTableRF,fullfile(pwd,'errorsTableRF.xls'),'WriteRowNames',true');

save(fullfile(pwd,'resultadosArtículo'));


%% Final results for 44 utterances
% clc
for indexFinal = 1:44
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix RF:');
    disp(sum(confMatrixRFStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixRFStc.(testToDo(indexFinal)),3);
    disp(['Specificity RF: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity RF: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy RF: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k


%% Final results for Table 2. Optimal utterance lists
% clc
for indexFinal = 45:57
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix RF:');
    disp(sum(confMatrixRFStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixRFStc.(testToDo(indexFinal)),3);
    disp(['Specificity RF: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity RF: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy RF: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k


%% Final results for Table 3. Utterances and classifiers used in the HN Score (7 + Sel)
% clc
for indexFinal = 57:length(testToDo)
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix RF:');
    disp(sum(confMatrixRFStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixRFStc.(testToDo(indexFinal)),3);
    disp(['Specificity RF: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity RF: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy FR: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k























