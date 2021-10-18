clc;
clear;
warning off

%%

File_name = {
    'deng-reads-RawCount.csv'
    };


%%
fileID = fopen('Result.txt','wt');

fprintf(fileID,'%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t\n',...
    'Method','with','Model','File_name','err','acc_train','acc','VRC','Silhouette','F1');

%%
for i = 1:length(File_name)
    
    % Loading the data
    file_name = File_name{i}; %
    
    opts = detectImportOptions(file_name,'NumHeaderLines',0);
    Data = readtable(file_name,opts);
    
    file_name = file_name(1:end-4);
    
    %% Extracting Different Group
    words_row = Data.Properties.VariableNames;
    words_row = regexprep(words_row,'_\d+$','');
    Class_label = words_row(2:end);
    
    
    X = Data{:,2:end};
    
    %% Clean data
    YY(:,1) = Class_label;
    
    x_bal = X; %x_bal(:,pp);
    y_bal = YY; %y_bal(pp);
    
    String = {
        'alpha.contaminated', 'beta.contaminated', ...
        'delta.contaminated', 'Excluded', 'gamma.contaminated', 'miss', 'NA', ...
        'not applicable', 'unclassified', 'unknown', 'Unknown', 'zothers'};
    
    id_na = 0;
    for iii = 1:length(String)
        iddd_na = strcmp(y_bal,String{iii});
        id_na = iddd_na + id_na;
    end
    
    idx_na = [];
    if sum(id_na)~=0
        idx_na = find(id_na ==1);
        y_bal(idx_na) = [];
        x_bal(:,idx_na) = [];
    end
    
    idx_0 = sum(x_bal) == 0;
    idy_0 = sum(x_bal,2) ==0;
    
    y_bal(idx_0)   = [];
    x_bal(:,idx_0) = [];
    x_bal(idy_0)   = [];
    
    
    [count,label] = hist(categorical(y_bal),unique(y_bal));
    
    NoCell = length(y_bal);
    
    disp(['# Cell : ' num2str(NoCell)]);
    disp(['# Label: ' num2str(length(label))]);
    disp(['File Name : ' file_name]);
    
    %%
    
    r = randperm(size(x_bal,2));
    SampleSize = size(X,2);
    
    y_bal = y_bal(r);
    X_raw = x_bal(:,r);
    
    %% Applying some normalizaiton
    %  Ex; CPM, Log CPM, Norm CPM and Norm Log CPM
    
    f_cpm = @(x)(x./sum(x))*1e6;
    
    X_cpm = f_cpm(X_raw);
    
    X_log = X_cpm;
    X_log(find(X_log==0)) = 1e-5;
    X_log = log(abs(X_log));
    
    
    X_norm_cpm = normalize(X_cpm','range')';
    
    X_norm_log = normalize(X_log','range')';
    %%
    
    X_pre_psd = {X_raw, X_cpm, X_log, X_norm_cpm, X_norm_log};
    
    %% Split data into Train (80%) and Test (20%)
    cv = cvpartition(size(X_raw,2),'HoldOut',0.20);
    idx = cv.test;
    
    %%
    Clf_name = {'RF', 'KNN','SVM'};
    X_psd_name = {'Raw', 'CPM', 'Log-CPM', 'Norm-CPM', 'Norm-Log-CPM'};
    %%
    for i_psd = 1:3 %length(X_pre_psd)
        
        figure;
        set(gcf,'position',[200,200,1000,500]);
        
        for K_method = [1 2]
            
            aa = X_pre_psd{i_psd};
            
            switch K_method
                case 1
                    
                    States = 'AfterPSD';
                    [A,~] = scPSD(aa,[]);
                    
                    subplot(1,2,1)
                    
                case 2
                    
                    States = 'BeforePSD';
                    A = aa;
                    
                    subplot(1,2,2)
            end
            
            disp(States);
            
            X_psd = A;
            %%
            data_complexity_F1 = complexity(X_psd,y_bal);
            disp(['Complexity: ' num2str(data_complexity_F1)]);
            %% TSNE
            
            disp('t-SNE ...')
            
            Y_tsne = tsne(X_psd','NumDimensions',2,'Algorithm',...
                'barneshut','Distance','euclidean');
            
            clr = jet(length(unique(y_bal)));
            
            gscatter(Y_tsne(:,1),Y_tsne(:,2),y_bal,clr);
            
            xlabel('t-SNE 1');
            ylabel('t-SNE 2');
            
            disp('done!')
            
            %% Cluter tendency
            
            Y_clust_real =zeros(size(y_bal));
            
            for ii = 1:length(label)
                id_clust = strcmp(y_bal,label{ii});
                Y_real   = id_clust*ii;
                Y_clust_real = Y_real + Y_clust_real;
            end
            
            eva_vrc = evalclusters(X_psd',Y_clust_real,'CalinskiHarabasz');
            VRC = eva_vrc.CriterionValues;
            
            disp(['VRC:' num2str(VRC)]);
            
            eva_silhou = evalclusters(X_psd',Y_clust_real,'silhouette');
            silhouette = eva_silhou.CriterionValues;
            
            disp(['silhouette:' num2str(silhouette)]);
            
            title([States '-' X_psd_name{i_psd} ': VRC = ' num2str(VRC) ...
                ', SS = ' num2str(silhouette) ', F1 = ' num2str(data_complexity_F1)]);
            
            save(['TSNE_' States '_' X_psd_name{i_psd} '_' file_name],...
                'Y_tsne','y_bal','data_complexity_F1','silhouette','VRC');
            %%
            if K_method == 1
                [XTrain,cor] = scPSD(aa(:,~idx),[]);
                [XTest,~]  = scPSD(aa(:,idx),cor);
            else
                XTrain = X_psd(:,~idx);
                XTest  = X_psd(:,idx);
            end
            
            YTrain = y_bal(~idx);
            YTest  = y_bal(idx) ;
            %% Machine Learning Method
            
            % 1: RF, 2: KNN, 3: SVM
            for n = [1 2 3]
                
                disp(['Classifier: ', Clf_name{n}])
                
                optimize = false;
                
                [err,acc_train,acc,sen,spe,order,cm,model] = F_classifier(XTrain',YTrain, ...
                    XTest',YTest,n,optimize);
                
                disp(['Acc: ', num2str(acc)])
                
                result.err = err; result.acc_train = acc_train; result.acc = acc; result.sen = sen;
                result.spe = spe; result.order = order; result.cm = cm; result.model = model;
                
                fprintf(fileID,'%s\t %s\t %s\t %s\t %2.4f\t %2.4f\t %2.4f\t %2.4f\t %2.4f\t %2.4f\t\n',...
                    States,X_psd_name{i_psd},Clf_name{n},file_name(1:end-9),err,...
                    acc_train,acc,VRC,silhouette,data_complexity_F1);
                
                save(['MachineModel_Result_' States '_' X_psd_name{i_psd} '_' ...
                    Clf_name{n} '_' file_name(1:end-9)], 'idx','r','model','cm','order');
                
                disp('done!')
            end
            
            
        end
        
    end
end
fclose(fileID);

%%

