clc;
clear;
warning off

%%

file_name = 'deng-reads-RawCount.csv'; 

opts = detectImportOptions(file_name,'NumHeaderLines',0);
Data = readtable(file_name,opts);

file_name = file_name(1:end-4);



%% Extracting Different Group
words_row = Data.Properties.VariableNames;

words_row = regexprep(words_row,'_\d+$','');


Class_label = words_row(2:end);


X = Data{:,2:end};

%%
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

disp(['No Cell : ' num2str(NoCell)]);
disp(['No Label: ' num2str(length(label))]);
disp(['File Name : ' file_name]);

%%

X_psd_orginal = scPSD(x_bal);

% Set the number of scPSD repetition for ordering Analysis
No_Iteration = 100;
%%
for i_s = 1:No_Iteration
    
    disp(i_s)
    
    r = randperm(size(x_bal,1));
    
    X_raw = x_bal(r,:);
    
    [X_psd,~] = scPSD(X_raw,[]);
    
    id_orginal = sort(r,'ascend');
    X_psd = X_psd(id_orginal,:);
    
    XX_PSD{i_s} = X_psd;
end

%%

for col = 1:size(X_psd_orginal,2)
    for i_f = 1:No_Iteration
        X_Iterorder(:,i_f) = XX_PSD{i_f}(:,col);
    end
    XX_allCol{col} = X_Iterorder;
end

%%
X_rmse = [];
for ic = 1:col
    X_rmse = [X_rmse sqrt(sum((XX_allCol{ic}-mean(XX_allCol{ic},2)).^2,2))./100];
end
%%
clims = [0 0.2];
imagesc(X_rmse,clims)
colorbar
colormap(flipud(bone))
xlabel('cells')
ylabel('genes/features')
title('Root-mean-square deviation')
%%









