% -------------------------------------------------------------------------
% clasificationHypernasality.m
% -------------------------------------------------------------------------
% DESCRIPTION:
% This script makes the hypernasality clasification.
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
sNamesVariables = tableGlobal.Properties.VariableNames(2:end);
sNamesVariablesShort = unique(cellfun(@(x) extractAfter(x(1:end),'_'),...
                sNamesVariables,'UniformOutput',false));     
testToDo = string(sNamesVariablesShort);
subTest = sNamesVariablesShort;
% Adding Selection and Global tests
testToDo = [testToDo,"T_Selection","T_Global"];
subTest{end+1} = ["T2_ka","T2_ki","T2_pi","T6_ADavidLeDuele","T3f","T5_dedo","T6_LaliyLunaLeen"];
subTest{end+1} = [string(sNamesVariablesShort)];


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

% False if dont want to test with SVM, RF, or ANN
testSVM = true;
testRF = true;
testANN = true;

numFeaturesRedux = 20;
featuresNamesSelectedStc = struct();
wordsSentencesSelectedStc = struct();

confMatrixSVMStc = struct();
confMatrixRFStc = struct();
confMatrixannStc = struct();

sNamesErroresSVMStc = struct();
sNamesErroresRFStc = struct();
sNamesErroresANNStc = struct();

for indexTest = 1:length(testToDo)
    tic
    % Cada prueba se hace por separado
    disp(['Performing test ' num2str(indexTest) ': '  char(testToDo(indexTest))]);
    
    sNamesErroresRFStc.(testToDo(indexTest)) = [];
    sNamesErroresSVMStc.(testToDo(indexTest)) = [];
    sNamesErroresANNStc.(testToDo(indexTest)) = [];
    featuresNamesSelectedStc.(testToDo(indexTest)) = [];
    wordsSentencesSelectedStc.(testToDo(indexTest)) = [];
    
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
        
        if contains(testToDo(indexTest),["T_Selection","T_Global"])
            wordsSentencesSelectedStc.(testToDo(indexTest)) = [wordsSentencesSelectedStc.(testToDo(indexTest)) ...
                sort(extractAfter(featuresNames(idx(1:numFeaturesRedux)),'_'))];
        end % if
        
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
        if testSVM       
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
        end % if testSVM
        
        % RF
        if testRF
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
        end % if testRF
        
        % ANN
        if testANN
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

            net = patternnet([1024 1024 1024 1024]);

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
        end % if testRF
        
    end % for k
    
    toc
end % for indexTest
save(fullfile(pwd,'resultadosArtículo'));


%% Creamos una tabla con todos los usuarios y las pruebas que han pasado
tablaErroresSVM = zeros(length(patientNames),length(testToDo));
tablaErroresRF = zeros(length(patientNames),length(testToDo));
tablaErroresANN = zeros(length(patientNames),length(testToDo));

for indexErrTest = 1:length(testToDo)
    try
        errorsAux = sNamesErroresSVMStc.(testToDo(indexErrTest));
        indexErr = contains(patientNames,errorsAux);
        
        tablaErroresSVM(indexErr,indexErrTest) = 1;
    catch
    end % try
    
    try
        errorsAux = sNamesErroresRFStc.(testToDo(indexErrTest));
        indexErr = contains(patientNames,errorsAux);
        
        tablaErroresRF(indexErr,indexErrTest) = 1;
    catch
    end % try
    
    try
        errorsAux = sNamesErroresANNStc.(testToDo(indexErrTest));
        indexErr = contains(patientNames,errorsAux);
        
        tablaErroresANN(indexErr,indexErrTest) = 1;
    catch
    end % try
end % for k

errorsTableSVM = array2table(tablaErroresSVM,...
    'RowNames',patientNames,'VariableNames',testToDo');
errorsTableRF = array2table(tablaErroresRF,...
    'RowNames',patientNames,'VariableNames',testToDo');
errorsTableANN = array2table(tablaErroresANN,...
    'RowNames',patientNames,'VariableNames',testToDo');

writetable(errorsTableSVM,fullfile(pwd,'errorsTableSVM.xls'),'WriteRowNames',true');
writetable(errorsTableRF,fullfile(pwd,'errorsTableRF.xls'),'WriteRowNames',true');
writetable(errorsTableANN,fullfile(pwd,'errorsTableANN.xls'),'WriteRowNames',true');

save(fullfile(pwd,'resultadosArtículo'));


%% Final results
% clc
for indexFinal = 1:length(testToDo)
    disp(' ');
    disp(['-----Test ' char(testToDo(indexFinal)) '-----']);
    try
        disp('Selected features');
        tabulate(sort(featuresNamesSelectedStc.(testToDo(indexFinal))'))
        disp(' ');
        if contains(testToDo(indexFinal),["T_Selection","T_Global"])
            disp('Words/Sentences selected');
            tabulate(wordsSentencesSelectedStc.(testToDo(indexFinal)))
            disp(' ');
        end % if contains
    catch
    end % try
    
    try
        disp('Confusion Matrix SVM:');
        disp(sum(confMatrixSVMStc.(testToDo(indexFinal)),3))
        cMatrix = sum(confMatrixSVMStc.(testToDo(indexFinal)),3);
        disp(['Specificity SVM: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
        disp(['Sensitivity SVM: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
        disp(['Accuracy SVM: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
        disp(' ');
    catch
    end % try
    
    try
        disp('Confusion Matrix RF:');
        disp(sum(confMatrixRFStc.(testToDo(indexFinal)),3))
        cMatrix = sum(confMatrixRFStc.(testToDo(indexFinal)),3);
        disp(['Specificity RF: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
        disp(['Sensitivity RF: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
        disp(['Accuracy RF: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
        disp(' ');
    catch
    end % try
    
    try
        disp('Confusion Matrix ANN:');
        disp(sum(confMatrixANNStc.(testToDo(indexFinal)),3))
        cMatrix = sum(confMatrixANNStc.(testToDo(indexFinal)),3);
        disp(['Specificity ANN: ' num2str( cMatrix(2,2) / sum(cMatrix(2,:)) ) ]);
        disp(['Sensitivity ANN: ' num2str( cMatrix(1,1) / sum(cMatrix(1,:)) ) ]);
        disp(['Accuracy ANN: ' num2str( (cMatrix(1,1)+cMatrix(2,2))/sum(cMatrix(:)) ) ]);
        disp(' ');
    catch
    end % try
end % for k






















