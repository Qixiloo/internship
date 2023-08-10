
from sklearn.metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import platform
if platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def metrics(actual, predict):
    print("#################################")
    print('confusion_matrix\n', confusion_matrix(actual, predict))

    print('accuracy_score', accuracy_score(actual, predict))

    print('precision_score', precision_score(actual, predict))

    print('recall_score', recall_score(actual, predict))
    print('f1_score', f1_score(actual, predict))
    print("#################################")

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(confusion_matrix(actual, predict)), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()





def metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob, step: float = 0.02, start:float = 0., end:float = 1., thres:float = None):
    f1_res = {}
    if thres:
        thres_opt = thres
    else:
        thres_range = np.arange(start, end, step)
        for i in range(len(thres_range)):
            y_train_predict = (y_train_predict_prob > thres_range[i])
            f1 = np.round(f1_score(y_train, y_train_predict), 2)
            f1_res[i] = f1
        thres_opt = thres_range[sorted(f1_res.keys(), key = lambda x: f1_res[x], reverse=True)][0]
    y_train_predict = (y_train_predict_prob > thres_opt)
    y_test_predict = (y_test_predict_prob > thres_opt)
    print('Optimal thres is ', thres_opt)
    print('--------训练集--------')
    metrics(y_train, y_train_predict)
    print('--------测试集--------')
    metrics(y_test, y_test_predict)
    print('\n')

def __metrics_plot(thres_range, f1s, precisions, recalls, accuracys, roc, thres_opt, fpr, tpr, type):
    plt.close()
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes = axes.flatten()
    # fig.tight_layout(pad=7.)
    fig.suptitle(f'{type} thres_opt ={thres_opt}', fontsize=20)
    axes[0].plot(thres_range, [np.round(x,2)*100 for x in precisions], label='precision_score', linewidth='2')
    axes[0].plot(thres_range, [np.round(x,2)*100 for x in recalls], label='recall_score', linewidth='2')
    axes[0].axvline(x=thres_opt, ymin=0, ymax=1, linewidth='2', color='red', linestyle='--')
    #A high F1 score indicates a good balance between precision and recall, 
    # meaning the model is making accurate positive predictions while also capturing a high percentage of actual positive instances.
    axes[0].text(thres_opt + 0.05, 1, "Achieve highest F1", fontsize=7)
    axes[0].set_ylim(0, 120)
    axes[0].set_ylabel("%", fontsize=10)
    axes[0].set_xlabel("threshold", fontsize=10)
    axes[0].legend(fontsize=10)


    axes[1].plot(thres_range, [np.round(x,2)*100 for x in f1s], label='F1 Score', linewidth='2')
    #axes[0,1].plot(thres_range, [np.round(x,2)*100 for x in accuracys], label='Accuracy', linewidth='2')
    axes[1].axvline(x=thres_opt, ymin=0, ymax=1, linewidth='2', color='red', linestyle='--')
    axes[1].text(thres_opt + 0.05, 1, "Achieve highest F1", fontsize=7)
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel("%", fontsize=10)
    axes[1].set_xlabel("threshold", fontsize=10)
    axes[1].legend(fontsize=10)


    #axes[1,0].plot(thres_range, [np.round(x,2)*100 for x in f1s], label='F1 Score', linewidth='2')
    axes[2].plot(thres_range, [np.round(x,2)*100 for x in accuracys], label='Accuracy', linewidth='2')
    axes[2].axvline(x=thres_opt, ymin=0, ymax=1, linewidth='2', color='red', linestyle='--')
    axes[2].text(thres_opt + 0.05, 1, "Achieve highest F1", fontsize=7)
    axes[2].set_ylim(0, 120)
    axes[2].set_ylabel("%", fontsize=10)
    axes[2].set_xlabel("threshold", fontsize=10)
    axes[2].legend(fontsize=10)


    axes[3].plot([np.round(x,2)*100 for x in fpr], [np.round(x,2)*100 for x in tpr], label='ROC', linewidth='2')
    axes[3].set_xlabel("FPR %", fontsize=10)
    axes[3].set_ylabel("TPR(Recall) %", fontsize=10)
    axes[3].legend(fontsize=10)
    plt.show()
    

def metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob, step: float = 0.02, start:float = 0., end:float = 1.,thres:float = None):
   
    thres_range = np.arange(start, end, step)
    
    f1s = []
    f1_res = {}
    precisions = []
    recalls = []
    accuracys = []
    if thres:
        thres_opt = thres
    else:
        for i in range(len(thres_range)):
            y_train_predict = (y_train_predict_prob > thres_range[i])
            f1 = np.round(f1_score(y_train, y_train_predict), 2)
            f1_res[i] = f1
            f1s.append(f1)
            #y_train_predict = (y_train_predict_prob > thres_range[i])
            #f1s.append(f1_score(y_train, y_train_predict))
            precision = precision_score(y_train, y_train_predict)
            if not precision:
                precisions.append(np.nan)
            else:
                precisions.append(precision)
            recall = recall_score(y_train, y_train_predict)
            if not recall:
                recalls.append(np.nan)
            else:
                recalls.append(recall)
            accuracys.append(accuracy_score(y_train, y_train_predict))
    thres_opt = thres_range[sorted(f1_res.keys(), key = lambda x: f1_res[x], reverse=True)][0]
    roc = roc_auc_score(y_train, y_train_predict_prob)
    fpr, tpr, thresholds= roc_curve(y_train, y_train_predict_prob)
    #opt_idx = np.argmax(f1s)
    #thres_opt = start + opt_idx*step
    print('训练集 Optimal thres is ', thres_opt)
    __metrics_plot(thres_range, f1s, precisions, recalls, accuracys, roc, thres_opt, fpr, tpr, '训练集')
    print("*********************************")
    

    f1s = []
    precisions = []
    recalls = []
    accuracys = []
    for i in range(len(thres_range)):
        y_test_predict = (y_test_predict_prob > thres_range[i])
        f1s.append(f1_score(y_test, y_test_predict))
        precision = precision_score(y_test, y_test_predict)
        if not precision:
            precisions.append(np.nan)
        else:
            precisions.append(precision)
        recall = recall_score(y_test, y_test_predict)
        if not recall:
            recalls.append(np.nan)
        else:
            recalls.append(recall)
        accuracys.append(accuracy_score(y_test, y_test_predict))
    roc = roc_auc_score(y_test, y_test_predict_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predict_prob)
    #used to draw the ROC-AUC curve
    #opt_idx = np.argmax(f1s)
    #thres_opt = start + opt_idx*step
    #print('测试集 Optimal thres is ', thres_opt)
    __metrics_plot(thres_range, f1s, precisions, recalls, accuracys, roc, thres_opt, fpr, tpr, '测试集')
    print("*********************************")


