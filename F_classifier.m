function [err,acc_train,acc,sen,spe,order,cm,model] = F_classifier(xtrain,ytrain,xtest,ytest,n,optimize)

XTrain = xtrain;
YTrain = ytrain;

XTest = xtest;
YTest = ytest;

auto_optimize = optimize;

%'randomsearch' 'gridsearch' 'bayesopt'
Optimizer = 'bayesopt';
Max_iteration = 30;
ShowPlots = false;
UseParallel = true;

if auto_optimize
    % If no pool, do not create new one.
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool(2);
    end
end


switch n
    
    
    case 1 % Random Forest
        
        if auto_optimize
            t = templateTree();
            Model = fitcensemble(XTrain,YTrain,'OptimizeHyperparameters','all','Learners',t, ...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
        else
            
            t = templateTree('MaxNumSplits',38,'MinLeafSize',2,...
                'SplitCriterion','gdi','NumVariablesToSample',242); % n = size(XTrain,1); m_max = floor(log(n - 1)/log(3)); max split is 3 ^m_max.
            numTrees = 61;
            
            Method = 'Bag';
            Model = fitcensemble(XTrain,YTrain,'Learners',t, ...
                'NumLearningCycles',numTrees, ...
                'Method',Method); %,'LearnRate',learnRate);
            
        end
        
        
    case 2 % KNN
        
        if auto_optimize
            Model = fitcknn(XTrain,YTrain,'OptimizeHyperparameters','all',...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
        else
            NumNeighbors = 8;
            Distance = 'euclidean';
            BreakTies = 'nearest';
            Model = fitcknn(XTrain,YTrain,'NumNeighbors',NumNeighbors,...
                'NSMethod','exhaustive','Distance',Distance,...
                'BreakTies',BreakTies,'Standardize',true,'DistanceWeight','inverse');
            
        end
        
        
    case 3 % Multi Class SVM
        
        if auto_optimize
            t = templateSVM('SaveSupportVectors',true,'BoxConstraint',1,'KernelScale','auto');
            Model = fitcecoc(XTrain,YTrain,'OptimizeHyperparameters',{'Coding','Standardize','KernelScale','PolynomialOrder','KernelFunction'},'Learners',t, ...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,...
                'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
            
        else
            
            t = templateSVM('Standardize',false,'SaveSupportVectors',true,'BoxConstraint',1,'KernelFunction','linear');
            Model = fitcecoc(XTrain,YTrain,'Learners',t,'Coding','onevsall');
            
        end
        
end



CVModel = crossval(Model, 'kfold',5);
pred_train = kfoldPredict(CVModel);

[Class,~] = unique(YTrain);

cp_train = classperf(YTrain, pred_train);

err = cp_train.ErrorRate;
acc_train = cp_train.CorrectRate;

pred_test = predict(Model,XTest);

cp_test = classperf(YTest, pred_test);
acc =  cp_test.CorrectRate;
sen =  cp_test.Sensitivity;
spe =  cp_test.Specificity;


[cm,order] = confusionmat(YTest,pred_test);

model = Model;

end
