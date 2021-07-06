% -------------------------------------------------------------------------
% clasificationHypernasality.m
% -------------------------------------------------------------------------
% DESCRIPTION:
% This script makes the hypernasality clasification with SVM classifier.
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
base1 = "T5_dedo";
base2 = "T3f";
base3 = "T2_pa";
base4 = "T6_SusiSaleSola";

testToDo = [testToDo,"Table2_dedo","Table2_dedo_F","Table2_dedo_F_pa","Table2_dedo_F_pa_Susi",...
    "Table2_add_T2_pi","Table2_add_T2_ka","Table2_add_T2_ki","Table2_add_T5_pez",...
    "Table2_add_T6_ADavidLeDuele","Table2_add_T6_AlGatodeAgata","Table2_add_T2_ta","Table2_add_T2_ti"];

subTest{end+1} = [base1];
subTest{end+1} = [base1,base2];
subTest{end+1} = [base1,base2,base3];
subTest{end+1} = [base1,base2,base3,base4];
subTest{end+1} = [base1,base2,base3,base4,"T2_pi"];
subTest{end+1} = [base1,base2,base3,base4,"T2_ka"];
subTest{end+1} = [base1,base2,base3,base4,"T2_ki"];
subTest{end+1} = [base1,base2,base3,base4,"T5_pez"];
subTest{end+1} = [base1,base2,base3,base4,"T6_ADavidLeDuele"];
subTest{end+1} = [base1,base2,base3,base4,"T6_AlGatodeAgata"];
subTest{end+1} = [base1,base2,base3,base4,"T2_ta"];
subTest{end+1} = [base1,base2,base3,base4,"T2_ti"];

% Table 3. Utterances and classifiers used in the HN Score (7 + Sel)
testToDo = [testToDo,"Table3_dedoFPa","Table3_T5_dedo","Table3_T2_pi",...
    "Table3_T6_ADavid","Table3_pez"];

subTest{end+1} = ["T5_dedo", "T3f", "T2_pa"];
subTest{end+1} = ["T5_dedo", "T3f", "T2_pa"];
subTest{end+1} = ["T5_dedo"];
subTest{end+1} = ["T2_pi"];
subTest{end+1} = ["T6_ADavidLeDuele"];
subTest{end+1} = ["T5_pez"];


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

confMatrixSVMStc = struct();
sNamesErroresSVMStc = struct();

for indexTest = 1:length(testToDo)
    tic
    % Cada prueba se hace por separado
    disp(['Performing test ' num2str(indexTest) ': '  char(testToDo(indexTest))]);
    
    sNamesErroresSVMStc.(testToDo(indexTest)) = [];
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
        
        % SVM
        dataSVMTrain = trainDataPCA(randTrain,:);
        classDataSVMTrain = classTrain(randTrain);
        sNamesTrainSVM = sNamesTrain(randTrain);
        dataSVMTest = testDataPCA(randTest,:);
        classDataSVMTest = classTest(randTest);
        sNamesTestSVM = sNamesTest(randTest);
        
        SVMModel = fitcsvm(dataSVMTrain,classDataSVMTrain,'KernelFunction','linear',...
            'ClassNames',[0,1],'Standardize',true);
        
        [labelClasificador,~] = predict(SVMModel,dataSVMTest);
        confusM = confusionmat(double(classDataSVMTest),labelClasificador);
        sNamesErrorSVM = sNamesTestSVM(double(classDataSVMTest)~=labelClasificador);
        
        confMatrixSVMStc.(testToDo(indexTest))(:,:,k) = confusM;
        sNamesErroresSVMStc.(testToDo(indexTest)) = ...
            [sNamesErroresSVMStc.(testToDo(indexTest)); sNamesErrorSVM];
        
    end % for k
    
    toc
end % for indexTest
save(fullfile(pwd,'resultadosArtículo'));


%% Creamos una tabla con todos los usuarios y las pruebas que han pasado
tablaErroresSVM = zeros(length(patientNames),length(testToDo));

for indexErrTest = 1:length(testToDo)
    errorsAux = sNamesErroresSVMStc.(testToDo(indexErrTest));
    indexErr = contains(patientNames,errorsAux);
    
    tablaErroresSVM(indexErr,indexErrTest) = 1;
end % for k

errorsTableSVM = array2table(tablaErroresSVM,'RowNames',patientNames,'VariableNames',testToDo');
writetable(errorsTableSVM,fullfile(pwd,'errorsTableSVM.xls'),'WriteRowNames',true');

save(fullfile(pwd,'resultadosArtículo'));


%% Final results for 44 utterances
% clc
for indexFinal = 1:44
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix SVM:');
    disp(sum(confMatrixSVMStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixSVMStc.(testToDo(indexFinal)),3);
    disp(['Specificity SVM: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity SVM: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy SVM: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
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
    
    disp('Confusion Matrix SVM:');
    disp(sum(confMatrixSVMStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixSVMStc.(testToDo(indexFinal)),3);
    disp(['Specificity SVM: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity SVM: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy SVM: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
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
    
    disp('Confusion Matrix SVM:');
    disp(sum(confMatrixSVMStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixSVMStc.(testToDo(indexFinal)),3);
    disp(['Specificity SVM: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity SVM: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy SVM: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k

















