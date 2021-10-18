function y = complexity(X,classes)

% classes are the cell types annotated in the dataset.
s_class = unique(classes);

F1 = [];
for i=1:length(s_class)-1
    
    idx_c1 = strcmp(classes,s_class{i});
    
    mu_1  = mean(X(:,idx_c1),2);
    var_1 = var(X(:,idx_c1),0,2);
    
    
    for j=i+1:length(s_class)
        
        idx_c2 = strcmp(classes,s_class{j});
        
        mu_2  = mean(X(:,idx_c2),2);
        var_2 = var(X(:,idx_c2),0,2);
        
        f_num = (mu_1-mu_2).^2;
        f_den = var_1 + var_2 ;
        
        f_den(f_den==0) = 1e-8;
        
        f = f_num./f_den;
        
        F1 = [F1 trapz(f)];
        
    end
    
end

% the higher, the easier classifiers

y = mean(F1); 

end


