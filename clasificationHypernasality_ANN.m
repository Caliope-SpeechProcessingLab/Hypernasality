% -------------------------------------------------------------------------
% clasificationHypernasality.m
% -------------------------------------------------------------------------
% DESCRIPTION:
% This script makes the hypernasality clasification with ANN classifier.
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
base1 = "T2_pi";
base2 = "T6_SiLlueveleLlevo";

testToDo = [testToDo,"Table2_T2_pi0","Table2_T2_pi","Table2_T2_pi_SiLlueveleLlevo",...
    "Table2_add_T2_ka","Table2_add_T5_dedo","Table2_add_T5_pez","Table2_add_T2_ti",...
    "Table2_add_T6_LaliyLunaLeen","Table2_add_T2_ki","Table2_add_T2_ta","Table2_add_T5_gafas",...
    "Table2_add_T3f","Table2_add_T6_ADavidLeDuele"];

subTest{end+1} = [base1];
subTest{end+1} = [base1];
subTest{end+1} = [base1,base2];
subTest{end+1} = [base1,base2,"T2_ka"];
subTest{end+1} = [base1,base2,"T5_dedo"];
subTest{end+1} = [base1,base2,"T5_pez"];
subTest{end+1} = [base1,base2,"T2_ti"];
subTest{end+1} = [base1,base2,"T6_LaliyLunaLeen"];
subTest{end+1} = [base1,base2,"T2_ki"];
subTest{end+1} = [base1,base2,"T2_ta"];
subTest{end+1} = [base1,base2,"T5_gafas"];
subTest{end+1} = [base1,base2,"T3f"];
subTest{end+1} = [base1,base2,"T6_ADavidLeDuele"];


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

confMatrixAnnStc = struct();
sNamesErroresANNStc = struct();

for indexTest = 1:length(testToDo)
    tic
    % Cada prueba se hace por separado
    disp(['Performing test ' num2str(indexTest) ': '  char(testToDo(indexTest))]);
    
    sNamesErroresANNStc.(testToDo(indexTest)) = [];
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
        
        % ANN
        dataANNTrain = trainData';
        dataANNTest = testData';
        
        classDataANNTrain = [~classTrain';classTrain'];   % [0;1] o [1;0]
        classDataANNTest = [~classTest';classTest'];
        
        randTrain = randperm(size(dataANNTrain,2));
        randTest = randperm(size(dataANNTest,2));
        
        dataANNTrain = dataANNTrain(:,randTrain);
        classDataANNTrain = classDataANNTrain(:,randTrain);
        sNamesTrainANN = sNamesTrain(randTrain);
        
        dataANNTest = dataANNTest(:,randTest);
        classDataANNTest = classDataANNTest(:,randTest);
        sNamesTestANN = sNamesTest(randTest);
        
        inputsTrain = dataANNTrain;
        targetsTrain = classDataANNTrain;
        
        net = patternnet([1024 1024 1024 1024]); % 1024
        
        % Set up Division of Data for Training, Validation, Testing
        rng('default'); net.divideParam.trainRatio = 80/100;
        rng('default'); net.divideParam.valRatio = 20/100;
        rng('default'); net.divideParam.testRatio = 0/100;
        
        % net.trainParam.showWindow = true;
        net.trainParam.showWindow = false;
        
        [trainedNet,~] = train(net,inputsTrain,targetsTrain);
        close;
        
        % Test the Network
        inputsTest = dataANNTest;
        claseGleasonDosTestConfMatrix = classDataANNTest(1,:)+2*classDataANNTest(2,:);
        
        outputsTest = trainedNet(inputsTest);
        
        [~,clasesOutputTest] = max(outputsTest);
        confusM = confusionmat(claseGleasonDosTestConfMatrix,clasesOutputTest);
        sNamesErrorANN = sNamesTestANN(double(claseGleasonDosTestConfMatrix)~=clasesOutputTest);
        
        confMatrixANNStc.(testToDo(indexTest))(:,:,k) = confusM;
        sNamesErroresANNStc.(testToDo(indexTest)) = ...
            [sNamesErroresANNStc.(testToDo(indexTest)); sNamesErrorANN];
        
    end % for k
    
    toc
end % for indexTest
save(fullfile(pwd,'resultadosArtículo'));


%% Creamos una tabla con todos los usuarios y las pruebas que han pasado
tablaErroresANN = zeros(length(patientNames),length(testToDo));

for indexErrTest = 1:length(testToDo)
    errorsAux = sNamesErroresANNStc.(testToDo(indexErrTest));
    indexErr = contains(patientNames,errorsAux);
    
    tablaErroresANN(indexErr,indexErrTest) = 1;
end % for k

errorsTableANN = array2table(tablaErroresANN,'RowNames',patientNames,'VariableNames',testToDo');
writetable(errorsTableANN,fullfile(pwd,'errorsTableANN.xls'),'WriteRowNames',true');

save(fullfile(pwd,'resultadosArtículo'));


%% Final results for 44 utterances
% clc
for indexFinal = 1:44
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix ANN:');
    disp(sum(confMatrixANNStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixANNStc.(testToDo(indexFinal)),3);
    disp(['Specificity ANN: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity ANN: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy ANN: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k


%% Final results for Table 2. Optimal utterance lists
% clc
for indexFinal = 45:length(testToDo)
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    disp('Selected features');
    tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
    disp(' ');
    
    disp('Confusion Matrix ANN:');
    disp(sum(confMatrixANNStc.(testToDo(indexFinal)),3))
    cMatrix = sum(confMatrixANNStc.(testToDo(indexFinal)),3);
    disp(['Specificity ANN: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
    disp(['Sensitivity ANN: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
    disp(['Accuracy ANN: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
    disp(' ');
end % for k






















