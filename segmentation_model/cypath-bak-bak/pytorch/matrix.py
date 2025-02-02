import torch

def confusion_matrix(y_true, y_pred, N=2):
    indices = N * y_true + y_pred
    m = torch.bincount(indices, minlength=N**2).reshape(N,N)
    return m

def metricsN(cm):
    cm = cm.type(torch.DoubleTensor)
    accuracy = cm.diag().sum() / (cm.sum() + 1e-15)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    dice = 2.0 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    miou = iou.mean()
    return dict(
        accuracy = accuracy,
        precision = precision,
        recall = recall,
        dice = dice,
        iou = iou,
        miou = miou,
    )

def metrics2d(cm):
    cm = cm.type(torch.DoubleTensor)
    accuracy = cm.diag().sum() / (cm.sum() + 1e-15)
    accuracy = accuracy.item()#.detach().cpu().numpy()
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    precision = precision[1].item()#.detach().cpu().numpy()
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    recall = recall[1].item()#.detach().cpu().numpy()
    dice = 2.0 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)
    dice = dice[1].item()#.detach().cpu().numpy()
    specificity = cm[0][0] / (cm[0][0]+cm[0][1]+1e-15)
    specificity = specificity.item()
    return dict(
        accuracy = accuracy,
        precision = precision,
        specificity = specificity,
        recall = recall,
        dice = dice,
    )

def metrics(cm):
    cm = cm.type(torch.DoubleTensor)
    d = cm.shape[0]
    D = {}
    if d == 2:
        return metrics2d(cm)
    else:
        for i in range(d):
            subcm = torch.zeros(2,2)
            # TP
            subcm[1][1] = cm[i][i]
            # TN
            for x in range(d):
                for y in range(d):
                    if x==i or y==i:
                        continue
                    subcm[0][0]+=cm[x][y]
            # FN
            for y in range(d):
                if y != i:
                    subcm[1][0]+=cm[i][y]
            #FP
            for x in range(d):
                if x != i:
                    subcm[0][1]+=cm[x][i]
            D[i] = metrics2d(subcm)
    return D

