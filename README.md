# Single Cell Omics Data Transformation(scPSD)
We present a novel pre-processing method (scPSD) inspired by power spectral density analysis to extract important information from large-scale single-cell omics data and enhance the separation of cellular phenotypes.
The scPSD is originally implemented in Matlab, and Python code and R package are available on @VafaeeLab (https://github.com/VafaeeLab). 

### MATLAB Files
The MATLAB files includ "main.m" running the scPSD on "deng-read" data and reporting the accuracy of different classifiers such as SVM, RF and KNN for both before and after scPSD transformation. The "scPSD.m" is the MATLAB code of the proposed single omics data transformation (scPSD). "complexity.m" calculates the complexity for multi-classes data. "OrderingAnalysis.m" investigates the feature ordering sensitivity of scPSD. While scPSD feature extraction is independent of initial random ordering of the features, the order should remain the same across all cells, samples or experiments. 
