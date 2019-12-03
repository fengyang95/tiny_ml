import numpy as np
def precision_recall_curve(y_true,pred_prob):
    probs=sorted(list(pred_prob),reverse=True)
    Rs=[]
    Ps=[]
    for i in range(1,len(probs)):
        thresh=probs[i]
        preds_p=np.where(pred_prob>=thresh)[0]
        preds_n=np.where(pred_prob<thresh)[0]
        TP=len(np.where(y_true[preds_p]==1)[0])
        FP=len(preds_p)-TP
        FN=len(np.where(y_true[preds_n]==1)[0])
        #TN=len(preds_n)-FN
        R=TP/(TP+FN)
        S=TP/(TP+FP)
        Rs.append(R)
        Ps.append(S)

    return np.array(Ps),np.array(Rs)

def roc_curve(y_true,pred_prob):
    probs=sorted(list(pred_prob),reverse=True)
    TPRs=[]
    FPRs=[]
    for i in range(1,len(probs)):
        thresh = probs[i]
        preds_p = np.where(pred_prob >=thresh)[0]
        preds_n = np.where(pred_prob <thresh)[0]
        TP = len(np.where(y_true[preds_p] == 1)[0])
        FP = len(preds_p) - TP
        FN = len(np.where(y_true[preds_n] == 1)[0])
        # TN=len(preds_n)-FN
        TN=len(preds_n)-FN
        TPRs.append(TP/(TP+FN))
        FPRs.append(FP/(TN+FP))
    return np.array(FPRs),np.array(TPRs)

def roc_auc_score(y_true,pred_prob):
    FPRs,TPRs=roc_curve(y_true,pred_prob)
    auc=0.
    for i in range(0,len(FPRs)-1):
        auc+=0.5*(FPRs[i+1]-FPRs[i])*(TPRs[i+1]+TPRs[i])
    return auc

if __name__=='__main__':
    y_true=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
    pred_prob=np.array([0.7,0.9,0.2,0.8,0.3,0.64,0.53,0.12,0.34,0.52,0.98,0.03,0.32,0.4,
                        0.8,0.21,0.01,0.67,0.32,0.08,0.05,0.8,0.34,0.8])

    import matplotlib.pyplot as plt
    Ps,Rs=precision_recall_curve(y_true,pred_prob)
    plt.plot(Rs,Ps,label='tinyml')

    from sklearn.metrics import precision_recall_curve as sklearn_pr_curve
    Ps,Rs,_=sklearn_pr_curve(y_true,pred_prob)
    plt.plot(Rs,Ps,label='sklearn')
    plt.legend()
    plt.title('PRC')
    plt.show()

    FPR,TPR=roc_curve(y_true,pred_prob)
    plt.plot(FPR,TPR,label='tinyml')
    print('tinyml_auc:',roc_auc_score(y_true,pred_prob))
    from sklearn.metrics import roc_curve as sklearn_roc_curve
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
    FPR,TPR,_=sklearn_roc_curve(y_true,pred_prob)
    plt.plot(FPR,TPR,label='sklearn')
    plt.legend()
    plt.title('ROC')
    plt.show()
    print('sklearn auc:',sklearn_roc_auc_score(y_true,pred_prob))
