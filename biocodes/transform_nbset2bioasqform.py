import json,time
import numpy as np
import pandas as pd
import os, subprocess
import argparse
from collections import OrderedDict
import operator

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
    test = json.load(reader)

qidDict = dict()
for multiQid in test: # Supports Multi-qid
    assert len(multiQid) == (24+4) # Please use the lateset version of QA datasets. All multiQids should have length of 24 + 4 (3 for Sub id)
    if not multiQid[:-4] in qidDict:
        qidDict[multiQid[:-4]] = [test[multiQid]]
    else :
        qidDict[multiQid[:-4]].append(test[multiQid])


entryList = []
entryListWithProb = []

for qid in qidDict:

    jsonList = []
    for jsonele in qidDict[qid]:
        jsonList += jsonele

    qidDf = pd.DataFrame().from_dict(jsonList)
    
    sortedDf = qidDf.sort_values(by='probability', axis=0, ascending=False)

    sortedSumDict = OrderedDict()
    sortedSumDictKeyDict = dict()

	    
    for index in sortedDf.index:
        text = sortedDf.ix[index]["text"]
        if text == "":
            pass
        elif len(text) > 100:
            pass
        elif text.lower() in sortedSumDictKeyDict:
            sortedSumDict[sortedSumDictKeyDict[text.lower()]] += sortedDf.ix[index]["probability"]
        else:
            sortedSumDictKeyDict[text.lower()] = text
            sortedSumDict[sortedSumDictKeyDict[text.lower()]] = sortedDf.ix[index]["probability"]        
    finalSorted = sorted(sortedSumDict.items(), key=operator.itemgetter(1), reverse=True)

    
    entry = {u"type":"factoid", 
        u"id":qid, # must be 24 chars
        u"ideal_answer":"Dummy",
        u"exact_answer":[[ans[0]] for ans in finalSorted[:5]],
        }
    entryList.append(entry)
    
    entryWithProb = {u"type":"factoid", 
        u"id":qid, # must be 24 chars
        u"ideal_answer":"Dummy",
        u"exact_answer":[ans for ans in finalSorted[:20]],
        }
    entryListWithProb.append(entryWithProb)
finalformat = {u'questions':entryList}
finalformatWithProb = {u'questions':entryListWithProb}

if os.path.isdir(args.output_path):
    outfilepath = os.path.join(args.output_path, "BioASQform_BioASQ-answer.json")
    outWithProbfilepath = os.path.join(args.output_path, "WithProb_BioASQform_BioASQ-answer.json")
else:
    outfilepath = args.output_path
    outWithProbfilepath = args.output_path+"_WithProb"

with open(outfilepath, "w") as outfile:
    json.dump(finalformat, outfile, indent=2)
with open(outWithProbfilepath, "w") as outfile_prob:
    json.dump(finalformatWithProb, outfile_prob, indent=2)
