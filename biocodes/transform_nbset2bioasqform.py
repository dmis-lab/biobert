import json,time
import numpy as np
import pandas as pd
import os, subprocess
import argparse

parser = argparse.ArgumentParser(description='Shape the answer')
parser.add_argument('--nbest_path', type=str,  help='location of nbest_predictions.json')
parser.add_argument('--output_path', type=str,  help='location of nbest_predictions.json')
args = parser.parse_args()

    
### Setting basic strings 
#### Info : This script is only for factoid question


#### Checking nbest_BioASQ-test prediction.json
if not os.path.exists(args.nbest_path):
    print("No file exists!\n#### Fatal Error : Abort!")
    raise

#### Reading Pred File
with open(args.nbest_path, "r") as reader:
    test=json.load(reader)

entryList=[]
for qid in test:
    ansList=[] # plain list
    qidDf=pd.DataFrame().from_dict(test[qid])
    ansList=qidDf.sort_values(by='probability', axis=0, ascending=False)['text'][:5].tolist()
    
    entry={u"type":"factoid", 
    #u"body":qas, 
    u"id":qid, # must be 24 char
    u"ideal_answer":["Dummy"],
    u"exact_answer":[[ans] for ans in ansList if ans != " "],
    # I think enough?
    }
    entryList.append(entry)
finalformat={u'questions':entryList}

with open(args.output_path+"/BioASQform_BioASQ-answer.json", "w") as outfile:
    json.dump(finalformat,outfile)

