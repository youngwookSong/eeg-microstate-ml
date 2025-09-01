import pandas as pd
import numpy as np
import numbers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import permutation_importance

from sklearn.utils.metaestimators import _safe_split

from sklearn.base import clone

from joblib import Parallel
from sklearn.utils.fixes import delayed

from itertools import combinations

from sklearn.feature_selection import SelectKBest, f_classif
from skfeature.function.similarity_based import fisher_score

from collections import Counter


class feature_selector():
        
    def __init__(self):
        self.loocv_outflag = False
        self.loocv_inflag = False
        self.loocv_filt_flag = False


    def filter_method(self, est, X, y, cv, meth='anova', feat_num=20, norm=False, ROC_sel=True, importance_sel=True, 
                      n_jobs=1, verbose=0, pre_dispatch="2*n_jobs"):
        
        subsets = {}
        filt_score = X.columns.values

        count_arr = []
        importance_mean_arr = []

        X_ = X.values
        y_ = y.values

        for _, loo_test in cv.split(X_, y_, None):
            if (len(loo_test)==1):
                self.loocv_filt_flag = True
                break

        if len(filt_score) < feat_num:
                feat_num = len(filt_score)

        rng = np.random.default_rng(42)

        for range_i in range(feat_num):

            importance_arr = []
            filt_count = np.zeros(len(filt_score))

            parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

            result = parallel(
                delayed(self.filt_scorer)(
                    est,
                    X_,
                    y_,
                    train,
                    test,
                    meth,
                    range_i,
                    norm,
                    ROC_sel,
                    importance_sel,
                )
                for train, test in cv.split(X_, y_, None)
            )
            results = self._aggregate_score_dicts(result)

            if importance_sel:

                for i in range(len(results['feature'])):
                    importance_count = np.zeros(len(filt_score))
                    importance_count[:] = np.nan
                    filt_count[results['feature'][i]] = filt_count[results['feature'][i]] + 1
                    importance_count[results['feature'][i]] = results['importance'][i].importances_mean
                    importance_arr.append(abs(importance_count))

                importance_mean_arr.append(np.nanmean(importance_arr, axis=0))
            else:
                for i in range(len(results['feature'])):
                    filt_count[results['feature'][i]] = filt_count[results['feature'][i]] + 1
                
            count_arr.append(filt_count)
            
            if self.loocv_filt_flag:
                acc_score = np.array(results['accuracy'])
                del_arr1 = np.nonzero(y_ == 0)
                del_arr2 = np.nonzero(y_ == 1)
                sen = np.nanmean(np.delete(acc_score, del_arr1))
                spec = np.nanmean(np.delete(acc_score, del_arr2))
                bacc = (sen+spec)/2

                y_true_all  = np.array([np.ravel(v)[0] for v in results['y_true']]) 
                y_pred_all  = np.array([np.ravel(v)[0] for v in results['y_pred']]) 
                if 'y_score' in results and all(v is not None for v in results['y_score']):
                    y_score_all = np.array([np.ravel(v)[0] for v in results['y_score']])
                else:
                    y_score_all = np.full_like(y_true_all, np.nan, dtype=float)


                acc_point = float(np.mean(y_true_all == y_pred_all)) 
                lo, hi, acc_p = self._bootstrap_ci_accuracy(y_true_all, y_pred_all, 
                                                    B=5000, alpha=0.05, random_state=42, alternative='two-sided', p0=0.5)
                
                acc_ci = (lo, hi)                                 

                try:
                    f1 = f1_score(y_true_all, y_pred_all, average='binary', zero_division=0)
                except Exception:
                    f1 = np.nan

                try:
                    mask = ~np.isnan(y_score_all)
                    if ROC_sel and (mask.sum() > 1) and (len(np.unique(y_true_all[mask])) == 2):
                        auc = roc_auc_score(y_true_all[mask], y_score_all[mask])
                    else:
                        auc = np.nan
                except Exception:
                    auc = np.nan

                subsets[range_i] = {
                            "accuracy": np.mean(results['accuracy']),
                            "balanced_accuracy": bacc,
                            "sensitivity": sen,
                            "specificity": spec,
                            "accuracy_CI95": acc_ci,  
                            "accuracy_p": acc_p,    
                            "F1": f1,                
                            "ROC_AUC": auc            
                        }  
            else:

                if ROC_sel:
                    subsets[range_i] = {
                                    "accuracy": np.mean(results['accuracy']),
                                    "balanced_accuracy": np.mean(results['balanced_acc']),
                                    "sensitivity": np.mean(results['sensitivity']),
                                    "specificity": np.mean(results['specificity']),
                                    "ROC_AUC": np.mean(results['roc_auc']),
                                }
                else:
                    subsets[range_i] = {
                                    "accuracy": np.mean(results['accuracy']),
                                    "balanced_accuracy": np.mean(results['balanced_acc']),
                                    "sensitivity": np.mean(results['sensitivity']),
                                    "specificity": np.mean(results['specificity']),
                                }
                        
        count_arr = np.array(count_arr)
        

        self.filt_score_dp = pd.DataFrame(subsets).transpose()
        self.filt_count_dp = pd.DataFrame(data=count_arr, columns=filt_score)

        if importance_sel:
            importance_mean_arr = np.array(importance_mean_arr)
            self.filt_importance_dp = pd.DataFrame(data=importance_mean_arr, columns=filt_score)

        self.loocv_filt_flag = False


    def filt_scorer(self, est, X, y, train, test, meth, range_i, norm, ROC_sel, importance_sel):

        model = clone(est)

        result = {}

        X_train, y_train = _safe_split(model, X, y, train)
        X_test, y_test = _safe_split(model, X, y, test, train)


        if norm:
            X_std = np.std(X_train, axis=0)
            X_mean = np.mean(X_train, axis=0)

            X_train = (X_train-X_mean)/X_std
            X_test = (X_test-X_mean)/X_std

        if meth == 'anova':
            filt_ob = SelectKBest(f_classif, k=range_i+1).fit(X_train, y_train)
            X_feature = filt_ob.get_support()

        elif meth == 'fisher':
            fi_score = fisher_score.fisher_score(X_train, y_train)
            X_feature = np.argsort(fi_score)[::-1][:range_i+1]

        spl_X_train = X_train[:, X_feature]
        spl_X_test = X_test[:, X_feature]

        model.fit(spl_X_train, y_train)

        y_pred = model.predict(spl_X_test)                           
        y_score_scalar = np.nan                                             
        if ROC_sel:                                                         
            try:
                s = model.decision_function(spl_X_test)
                y_score_scalar = float(np.ravel(s)[0])
            except Exception:
                try:
                    proba = model.predict_proba(spl_X_test)
                    if proba.shape[1] == 2:
                        y_score_scalar = float(np.ravel(proba[:, 1])[0])
                except Exception:
                    y_score_scalar = np.nan

        

        confusionMatrix = confusion_matrix(y_test, y_pred)
        
        if self.loocv_filt_flag:
            
            if confusionMatrix.shape[0] == 1:
                acc = confusionMatrix[0,0]
            elif confusionMatrix.shape[0] == 2:
                acc = confusionMatrix[1,1]

            if importance_sel:

                result_im = permutation_importance(model, spl_X_train, y_train, n_repeats=100, random_state=42)

                result["feature"] = X_feature
                result["accuracy"] = acc
                result["importance"] = result_im

            else:
                result["feature"] = X_feature
                result["accuracy"] = acc

        else:
            TN = confusionMatrix[0,0]
            TP = confusionMatrix[1,1]
            FN = confusionMatrix[1,0]
            FP = confusionMatrix[0,1]

            acc = (TP+TN)/(TP+FP+FN+TN)
            sen = TP/(TP+FN)
            spec = TN/(TN+FP)
            b_acc = (sen+spec)/2

            if ROC_sel:
                y_scores = model.decision_function(spl_X_test)
                auc_score = roc_auc_score(y_test, y_scores)

                if importance_sel:

                    result_im = permutation_importance(model, spl_X_train, y_train, n_repeats=100, random_state=42)

                    result["feature"] = X_feature
                    result["accuracy"] = acc
                    result["balanced_acc"] = b_acc
                    result["sensitivity"] = sen
                    result["specificity"] = spec
                    result["roc_auc"] = auc_score
                    result["importance"] = result_im

                else:

                    result["feature"] = X_feature
                    result["accuracy"] = acc
                    result["balanced_acc"] = b_acc
                    result["sensitivity"] = sen
                    result["specificity"] = spec
                    result["roc_auc"] = auc_score

            else:

                if importance_sel:

                    result_im = permutation_importance(model, spl_X_train, y_train, n_repeats=100, random_state=42)

                    result["feature"] = X_feature
                    result["accuracy"] = acc
                    result["balanced_acc"] = b_acc
                    result["sensitivity"] = sen
                    result["specificity"] = spec
                    result["importance"] = result_im

                else:
                    result["feature"] = X_feature
                    result["accuracy"] = acc
                    result["balanced_acc"] = b_acc
                    result["sensitivity"] = sen
                    result["specificity"] = spec
        
        
        result["y_true"]  = np.array(y_test)                            
        result["y_pred"]  = np.array(y_pred)                               
        result["y_score"] = np.array([y_score_scalar]) if ROC_sel else None  

        return result

    
    def _aggregate_score_dicts(self, scores):
        return {
            key: np.asarray([score[key] for score in scores])
            if isinstance(scores[0][key], numbers.Number)
            else [score[key] for score in scores]
            for key in scores[0]
        }
    
    def _bootstrap_ci_accuracy(self, y_true, y_pred, B=5000, alpha=0.05, random_state=42, alternative='two-sided', p0=0.5):
        rng = np.random.default_rng(random_state)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        acc_point = float(np.mean(y_true == y_pred))
        vals = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, size=n)       
            vals[b] = np.mean(y_true[idx] == y_pred[idx]) 
        lo = float(np.percentile(vals, 2.5))
        hi = float(np.percentile(vals, 97.5))

        # ------p-value--------
        p_boot = None

        # Normal approximation using bootstrap SE
        se = float(np.std(vals, ddof=1))
        if se == 0.0:
            if alternative == 'greater':
                p_boot = 0.0 if acc_point > p0 else 1.0
            elif alternative == 'less':
                p_boot = 0.0 if acc_point < p0 else 1.0
            else:  # two-sided
                p_boot = 0.0 if acc_point != p0 else 1.0
        else:
            z = (acc_point - float(p0)) / se
            cdf = 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))
            if alternative == 'greater':
                p_boot = 1.0 - cdf
            elif alternative == 'less':
                p_boot = cdf
            else:  # two-sided
                p_boot = 2.0 * min(cdf, 1.0 - cdf)
        return (lo, hi, float(p_boot))