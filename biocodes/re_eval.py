import os
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
args = parser.parse_args()




testdf = pd.read_csv(args.answer_path, sep="\t", index_col=0)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)

pred = [preddf.iloc[i].tolist() for i in preddf.index]
pred_class = [np.argmax(v) for v in pred]
pred_prob_one = [v[1] for v in pred]




p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])
results = dict()
results["f1 score"] = f[1]
results["recall"] = r[1]
results["precision"] = p[1]
results["specificity"] = r[0]

for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))
