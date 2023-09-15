import sys
import os
import time
import datetime
import numpy as np
from sklearn import metrics
from tools.tools import *
import tensorflow as tf
from torch.nn import BCELoss
import torch

def test(config, model, test_type):
    print(
        str(datetime.datetime.now()) 
        + ' Starting testing the %s set with '%(test_type)
        + str(config['n_valid']) + ' subgraphs!'
        )

    # Start timer
    t_start = time.time()
    
    # load data
    if test_type == 'valid':
        valid_data = get_dataset(config['valid_dir'], config['n_valid'])
        n_test = config['n_valid']
        log_extension = 'validation'
    elif test_type == 'train':
        valid_data = get_dataset(config['train_dir'], config['n_train'])
        n_test = config['n_train']
        log_extension = 'training'

    # Load loss function
    loss_func = BCELoss()

    # Obtain predictions and labels
    for n in range(n_test):

        X, Ri, Ro, y = valid_data[n]

        if n == 0:
            preds = model([map2angle(X), Ri, Ro])
            # labels = y
            label__ = torch.tensor(y)
            label__=label__[:,None]

        else:	
            out = model([map2angle(X), Ri, Ro])
            labelsss = torch.tensor(y)
            labelsss=labelsss[:,None]
            preds  = torch.cat((preds, out), 0)
            label__ = torch.cat((label__, labelsss), 0)
            

    # labels = tf.reshape(labels, shape=(labels.shape[0],1))

    # # calculate weight for each edge to avoid class imbalance
    # weights = tf.convert_to_tensor(true_fake_weights(labels))
    print(preds.shape)
    print(label__.shape)
    loss = loss_func(preds, label__)

    # Log all predictons (takes some considerable time - use only for debugging)
    if config['log_verbosity']>=3 and test_type=='valid':	
        with open(config['log_dir']+'log_validation_preds.csv', 'a') as f:
            for i in range(len(preds)):
                f.write('%.4f, %.4f\n' %(preds[i],label__[i]))

    # Calculate Metrics
    # To Do: add 0.8 threshold and other possible metrics
    # efficency, purity etc.
    # labels = labels.numpy()
    # preds  = preds.numpy()

    #n_edges = labels.shape[0]
    #n_class = [n_edges - sum(labels), sum(labels)]
    thld=0.5
    TP = torch.sum((label__==1).squeeze() & 
                    (preds>thld).squeeze()).item()
    TN = torch.sum((label__==0).squeeze() & 
                    (preds<thld).squeeze()).item()
    FP = torch.sum((label__==0).squeeze() & 
                    (preds>thld).squeeze()).item()
    FN = torch.sum((label__==1).squeeze() & 
                    (preds<thld).squeeze()).item()            
    acc_5= (TP+TN)/(TP+TN+FP+FN)
    precision_5 = TP/(TP+FP) # also named purity
    recall_5    = TP/(TP+FN) # also named efficiency
    f1_5        = (2*precision_5*recall_5)/(precision_5+recall_5) 

    # fpr, tpr, _ = metrics.roc_curve(labels.astype(int),preds,pos_label=1 )
    # auc                = metrics.auc(fpr,tpr)

    # End timer
    duration = time.time() - t_start

    # Log Metrics
    with open(config['log_dir']+'log_'+log_extension+'.csv', 'a') as f:
        # f.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n' %(accuracy_5, auc, loss, precision_5, accuracy_3, precision_3, recall_3, f1_3, accuracy_5, precision_5, recall_5, f1_5, accuracy_7, precision_7, recall_7, f1_7, duration))
        f.write('%f, %f, %f, %f, %f, %d\n' %(acc_5, loss, precision_5, recall_5, f1_5, duration))

    # Print summary
    print(str(datetime.datetime.now()) + ': ' + log_extension+' Test:  Loss: %.4f,  Acc: %.4f,  Precision: %.4f -- Elapsed: %dm%ds' %(loss, acc_5*100, precision_5, duration/60, duration%60))

    del label__
    del preds
