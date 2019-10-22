import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--token_test_path', type=str,  help='')
parser.add_argument('--label_test_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--output_dir', type=str,  help='')
args = parser.parse_args()

def detokenize(golden_path, pred_token_test_path, pred_label_test_path, output_dir):
    
    """convert word-piece bioBERT-NER results to original words (CoNLL eval format)
        
    Args:
        golden_path: path to golden dataset. ex) NCBI-disease/test.tsv
        pred_token_test_path: path to token_test.txt from output folder. ex) output/token_test.txt
        pred_label_test_path: path to label_test.txt from output folder. ex) output/label_test.txt
        output_dir: path to output result will write on. ex) output/
        
    Outs:
        NER_result_conll.txt
    """
    # read golden
    ans = dict()
    ans['toks'] = list()
    ans['labels'] = list()
    lineNoCount=0
    with open(golden_path,'r') as in_:
        for line in in_:
            line = line.strip()
            if line == '':
                ans['toks'].append('[SEP]')
                lineNoCount+=1
                continue
            tmp = line.split()
            ans['toks'].append(tmp[0])
            ans['labels'].append(tmp[1])
    # len(ans['labels'])=ans['toks']-lineNoCount
    
    # read predicted
    pred = dict({'toks':[], 'labels':[]}) # dictionary for predicted tokens and labels.
    with open(pred_token_test_path,'r') as in_: #'token_test.txt'
        for line in in_:
            line = line.strip()
            pred['toks'].append(line)
            
    with open(pred_label_test_path,'r') as in_: #'label_test_3_epoch.txt'
        for line in in_:
            line = line.strip()
            if line in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. This will not be evaluated.
                pred['labels'].append('O')
                continue
            pred['labels'].append(line)
            
    if (len(pred['toks']) != len(pred['labels'])): # Sanity check
        print("Error! : len(pred['toks']) != len(pred['labels']) : Please report us")
        raise
    
    bert_pred = dict({'toks':[], 'labels':[]})
    for t, l in zip(pred['toks'],pred['labels']):
        if t in ['[CLS]','[SEP]']: # non-text tokens will not be evaluated.
            continue
        elif t[:2] == '##': # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_pred['toks'][-1] = bert_pred['toks'][-1]+t[2:] # append pieces
        else:
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(l)
    
    if (len(bert_pred['toks']) != len(bert_pred['labels'])): # Sanity check
        print("Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")
        raise
   
    if (len(ans['labels']) != len(bert_pred['labels'])): # Sanity check
        print(len(ans['labels']), len(bert_pred['labels']))
        print("Error! : len(ans['labels']) != len(bert_pred['labels']) : Please report us")
        raise
    
    with open(output_dir+'/NER_result_conll.txt', 'w') as out_:
        idx=0
        for ans_t in ans['toks']:
            if ans_t=='[SEP]':
                out_.write("\n")
            else :
                out_.write("%s %s-MISC %s-MISC\n"%(bert_pred['toks'][idx], ans['labels'][idx], bert_pred['labels'][idx]))
                idx+=1

detokenize(args.answer_path, args.token_test_path, args.label_test_path, args.output_dir)
