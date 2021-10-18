function [X_psd,cor] = scPSD(x,bb)

aa = x;

% Correlation Analysis
cor_x = corr(aa');
cor_x(isnan(cor_x)) = 1;

cor = cor_x; 

if ~isempty(bb)
    cor = bb;
end

aa_cor = cor*aa;

% Discrete Fourier Transformation
ps = abs(fft(abs(aa_cor)))/size(aa,1);

% Entropy Estimation
pro_ps = ps./sum(ps,1);
Aaa = pro_ps.*log2(pro_ps);
aaa = -(mean(Aaa,2)-Aaa);

% Scaling in [0,1]
bb = aaa;
A = normalize(bb','range')';

X_psd = A;

end