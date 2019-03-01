#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn import metrics
import math

target = sys.argv[3]
metric = sys.argv[4]
Ytest = pd.read_csv(sys.argv[1])[target]
predictions = pd.read_csv(sys.argv[2])[target]

if metric == 'MSE':
    print(metrics.mean_squared_error(Ytest, predictions))
elif metric == 'F1':
    print(metrics.f1_score(Ytest, predictions, average='macro'))
elif metric == 'MAE':
    print(metrics.mean_absolute_error(Ytest, predictions))
elif metric == 'ACC':
    print(metrics.accuracy_score(Ytest, predictions))
elif metric == 'NMI':
    print(metrics.normalized_mutual_info_score(Ytest, predictions))
elif metric == 'RMSE':
    print(math.sqrt(metrics.mean_squared_error(Ytest, predictions)))
