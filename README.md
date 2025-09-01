# eeg-microstate-ml
EEG microstate-based classification using machine learning in depressed adolescents with and without non-suicidal self-injury
(EEG Microstate–Based ML for Adolescent MDD / NSSI)

This repository contains a minimal, reproducible pipeline for EEG microstate feature selection and classification in adolescents with MDD, with/without NSSI.

# What’s inside
filter_method_fisher.py — core implementation (class feature_selector) for:
  
	Train-fold-only z-scoring and Fisher/ANOVA ranking (top-k scan)
  
	LOOCV evaluation with pooled out-of-sample predictions
  
	Metrics: accuracy, balanced accuracy, sensitivity, specificity, F1, AUC
  
	Accuracy 95% bootstrap CI (B=5,000; seed=42) and a bootstrap-normal p-value (two-sided vs 0.5)

  Outputs:
    
		filt_score_dp: performance vs. k (rows = k−1, columns = metrics)
    
		filt_count_dp: feature selection frequency across folds
    
		filt_importance_dp (optional): permutation importance means
    
		(See source for details.) 

sfs_NSSI_filter.ipynb — example notebook to reproduce analyses/figures from feature tables.
