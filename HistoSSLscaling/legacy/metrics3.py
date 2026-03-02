import sklearn
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
import numpy as np
import math
from prettytable import PrettyTable
import scipy.stats as st

def acc(y_true, y_pred):
    return skm.accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    return skm.f1_score(y_true, y_pred)

def precision(y_true, y_pred):
    return skm.precision_score(y_true, y_pred)

def recall(y_true, y_pred): # sensitivity
    return skm.recall_score(y_true, y_pred)

def specificity(y_true, y_pred):
    try:
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    except:
        return float('nan')
    return tn / (tn+fp+1e-15)

def negative_predictive_value(y_true, y_pred): #npv
    try:
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    except:
        return float('nan')
    return tn / (tn+fn+1e-15)

def auc_score(y_true, y_prob):
    # parameter estimation
    def roc_auc_ci(y_true, y_score, positive=1):
        # source : https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
        # explain: https://stats.stackexchange.com/questions/165033/how-to-interpret-95-confidence-interval-for-area-under-curve-of-roc
        N1 = np.sum(y_true == positive)
        N2 = np.sum(y_true != positive)
        if N1==0 or N1==len(y_true):
            return float('nan'), float('nan'), float('nan')
        AUC = skm.roc_auc_score(y_true, y_score)
        Q1 = AUC / (2 - AUC)
        Q2 = 2*AUC**2 / (1 + AUC)
        SE_AUC = math.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
        lower = AUC - 1.96*SE_AUC
        upper = AUC + 1.96*SE_AUC
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return 1.96*SE_AUC, lower, upper
    fpr, tpr, _ = skm.roc_curve(y_true, y_prob)
    roc_auc = skm.auc(fpr, tpr)
    ci = roc_auc_ci(y_true, y_prob)
    return dict(auc=roc_auc, ci=ci[0], lower=ci[1], upper=ci[2])

def auc_score_bootstrap(y_true, y_prob):
    main_fpr, main_tpr, _ = skm.roc_curve(y_true, y_prob)
    main_roc_auc = skm.auc(main_fpr, main_tpr)

    # https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
    # https://github.com/hirsch-lab/roc-utils/blob/64952e5c81733cc35f4dc6b9e14ef31df1e76611/roc_utils/_stats.py#L5

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    
    bootstrap_aucs=[]
    bootstrap_fprs=[]
    bootstrap_tprs=[]

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        fpr, tpr, _ = skm.roc_curve(y_true[indices], y_prob[indices])
        roc_auc = skm.auc(fpr, tpr)
        
        bootstrap_aucs.append(roc_auc)
        bootstrap_fprs.append(fpr)
        bootstrap_tprs.append(tpr)
    
    def mean_intervals(data, confidence=0.95, axis=None):
        """
        Compute the mean, the confidence interval of the mean, and the tolerance
        interval. Note that the confidence interval is often misinterpreted [3].
        References:
        [1] https://en.wikipedia.org/wiki/Confidence_interval
        [2| https://en.wikipedia.org/wiki/Tolerance_interval
        [3] https://en.wikipedia.org/wiki/Confidence_interval#Meaning_and_interpretation
        """
        confidence = confidence / 100.0 if confidence > 1.0 else confidence
        assert(0 < confidence < 1)
        a = 1.0 * np.array(data)
        n = len(a)
        # Both s=std() and se=sem() use unbiased estimators (ddof=1).
        m = np.mean(a, axis=axis)
        s = np.std(a, ddof=1, axis=axis)
        se = st.sem(a, axis=axis)
        t = st.t.ppf((1 + confidence) / 2., n - 1)
        #ci = np.c_[m - se * t, m + se * t]
        ci = se * t
        #ti = np.c_[m - s * t, m + s * t]
        #assert(ci.shape[1] == 2 and ci.shape[0] ==
        #    np.size(m, axis=None if axis is None else 0))
        #assert(ti.shape[1] == 2 and ti.shape[0] ==
        #    np.size(m, axis=None if axis is None else 0))
        #return m, ci, ti
        return dict(auc=m, ci=ci, lower=m-ci, upper=m+ci)
    res = mean_intervals(bootstrap_aucs)
    return res


def report(labels, preds, probs, classes, digits=4):
    # multi class metrics report
    # treat binary class also as multi class
    N_class = len(classes)
    if N_class < 2:
        assert 'Error, N_class must >= 2'
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    supports = np.bincount(labels, minlength=N_class)
    all_support = np.sum(supports)

    cm = skm.confusion_matrix(labels, preds, labels=[i for i in range(N_class)])

    report = dict()

    accuracy = acc(labels, preds)

    # compute binary metrics of each class

    marco_avg = dict()
    weighted_avg = dict()

    mstrs = ['f1-score', 'precision', 'recall', 'specificity', 'NPV']
    mfuns = [f1, precision, recall, specificity, negative_predictive_value]

    labels = label_binarize(labels, classes=list(np.arange(N_class+1)))[:,:-1]
    preds = label_binarize(preds, classes=list(np.arange(N_class+1)))[:,:-1]

    for pos_label in range(N_class):
        y_true = labels[:, pos_label]
        y_pred = preds[:, pos_label]
        y_prob = probs[:, pos_label]
        support = supports[pos_label]

        res = dict()

        #auc = auc_score(y_true, y_prob)
        auc = auc_score_bootstrap(y_true, y_prob)
      
        res['auc'] = round(auc['auc'], digits)
        res['auc-ci'] = round(auc['ci'], digits)
        res['auc-lower'] = round(auc['lower'], digits)
        res['auc-upper'] = round(auc['upper'], digits)

        for i in range(len(mstrs)):
            mstr = mstrs[i]
            mfun = mfuns[i]
            score = mfun(y_true, y_pred)
            res[mstr] = round(score, digits)
            if marco_avg.get(mstr) == None:
                marco_avg[mstr] = score / N_class
                weighted_avg[mstr] = score * support / all_support
            else:
                marco_avg[mstr] += score / N_class
                weighted_avg[mstr] += score * support / all_support

        res['support'] = supports[pos_label]
        report[str(pos_label)] = res

    marco_avg['support'] = all_support
    weighted_avg['support'] = all_support

    for mstr in mstrs:
        marco_avg[mstr] = round(marco_avg[mstr], digits)
        weighted_avg[mstr] = round(weighted_avg[mstr], digits)
    report['marco avg'] = marco_avg
    report['weighted avg'] = weighted_avg
    report['accuracy'] = round(accuracy, digits)

    # print cm
    print()
    print(cm)
    print()
    # print pretty auc report
    header = ['class', 'auc', '95%ci', 'auc_lower', 'auc_upper']
    t = PrettyTable(header)
    for i in range(N_class):
        # class_name = classes[i]
        i = str(i)
        res = report[i]
        row = [i, res['auc'], res['auc-ci'], res['auc-lower'], res['auc-upper']]
        t.add_row(row)
    print(t)

    # print pretty classification report
    header = ['class']
    header.extend(mstrs)
    header.append('support')
    t = PrettyTable(header)
    for i in range(N_class):
        i = str(i)
        res = report[i]
        row = [i]
        row.extend([res[mstr] for mstr in mstrs])
        row.append(res['support'])
        t.add_row(row)

    t.add_row(['' for _ in range(2+len(mstrs))])
    acc_row = ['accuracy']
    acc_row.extend(['' for _ in range(len(mstrs)-1)])
    acc_row.extend([report['accuracy'], all_support])
    t.add_row(acc_row)

    for avg in ['marco avg', 'weighted avg']:
        avg_row = [avg]
        avg_row.extend([report[avg][mstr] for mstr in mstrs])
        avg_row.append(report[avg]['support'])
        t.add_row(avg_row)

    print(t)
    return report

    
