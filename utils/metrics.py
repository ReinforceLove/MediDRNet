import torch
from sklearn import metrics
import numpy as np


def metrics_cal(pred,label):
    conf_matrix = metrics.confusion_matrix(label, pred)
    sklearn_accuracy = metrics.accuracy_score(label, pred)
    sklearn_precision = metrics.precision_score(label, pred, average=None,zero_division=0)
    sklearn_recall = metrics.recall_score(label, pred, average=None)
    sklearn_f1 = metrics.f1_score(label, pred, average=None,zero_division=0)
    measure_result = metrics.classification_report(label, pred)
    return sklearn_accuracy,sklearn_precision,sklearn_recall,sklearn_f1,conf_matrix,measure_result

if __name__=='__main__':
    pred = torch.randint(0,6,[40])
    label = torch.randint(0,6,[40])
    print(pred)
    print(label)


    conf_matrix = metrics.confusion_matrix(label, pred)
    sklearn_accuracy = metrics.accuracy_score(label, pred)
    sklearn_precision = metrics.precision_score(label, pred, average=None)
    sklearn_recall = metrics.recall_score(label, pred, average=None)
    sklearn_f1 = metrics.f1_score(label, pred, average=None)
    measure_result = metrics.classification_report(label, pred,digits=3)
    print(conf_matrix)
    print(sklearn_accuracy)
    print(sklearn_precision)
    print(sklearn_recall)
    print(sklearn_f1)
    print(measure_result)
