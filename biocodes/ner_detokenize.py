# Copyright DMIS Lab. BioBERT Authors. http://dmis.korea.ac.kr 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--token_test_path', type=str,  help='Path to token_test.txt from output folder. ex) output/token_test.txt')
parser.add_argument('--label_test_path', type=str,  help='Path to label_test.txt from output folder. ex) output/label_test.txt')
parser.add_argument('--answer_path', type=str, default=None,  help='Path to golden dataset. ex) NCBI-disease/test.tsv')
parser.add_argument('--output_dir', type=str,  help='Path to output result will write on. ex) output/')
parser.add_argument('--debug', action='store_true', help='Debug. Outputs NER_result_sent-debug.txt')
args = parser.parse_args()

def detokenize(pred_token_test_path, pred_label_test_path):
    """
    convert suub-word level BioBERT-NER results to full words and labels.
        
    Args:
        pred_token_test_path: path to token_test.txt from output folder. ex) output/token_test.txt
        pred_label_test_path: path to label_test.txt from output folder. ex) output/label_test.txt
    Outs:
        A dictionary that contains full words and predicted labels. 
    """

    # read predicted
    pred = {'toks':[], 'labels':[]} # dictionary for predicted tokens and labels.
    with open(pred_token_test_path,'r') as in_tok, open(pred_label_test_path,'r') as in_lab: #'token_test.txt'
        for lineIdx, (lineTok, lineLab) in enumerate(zip(in_tok, in_lab)):
            lineTok = lineTok.strip()
            pred['toks'].append(lineTok)
            
            lineLab = lineLab.strip()
            if lineLab in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. These will not be evaluated.
                pred['labels'].append('O')
                continue
            pred['labels'].append(lineLab)
        
    assert (len(pred['toks']) == len(pred['labels'])), "Error! : len(pred['toks'])(%s) != len(pred['labels'])(%s) : Please report us "%(len(pred['toks']), len(pred['labels']))
    
    bert_pred = {'toks':[], 'labels':[], 'sentence':[]}
    buf = []
    for t, l in zip(pred['toks'], pred['labels']):
        if t in ['[CLS]','[SEP]']: # non-text tokens will not be evaluated.
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(t) # Tokens and labels should be identical if they are [CLS] or [SEP]
            if t == '[SEP]':
                bert_pred['sentence'].append(buf)
                buf = []
            continue
        elif t[:2] == '##': # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_pred['toks'][-1] += t[2:] # append pieces to make a full-word
            buf[-1]+=t[2:]
        else:
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(l)
            buf.append(t)
    
    assert (len(bert_pred['toks']) == len(bert_pred['labels'])), ("Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")
    
    return bert_pred

def transform2CoNLLForm(golden_path, output_dir, bert_pred, debug):
    """
    Produce NER_result_conll.txt file that suits conlleval.pl
    Output : Line-seperated list of words, answer tags and predicted tags
    ex) 
    Association O-MISC O-MISC
    of O-MISC O-MISC
    ...
    """
    # read golden
    ans = {'toks':["[CLS]"], 'labels':["[CLS]"], 'sentence':[]}
    with open(golden_path,'r') as in_:
        buf = []
        #labelBuf=[]
        for lineIdx, line in enumerate(in_):
            if line.splitlines()[0] == '': # For safe line char
                ans['toks'].append('[SEP]')
                ans['toks'].append('[CLS]')
                ans['labels'].append('[SEP]')
                ans['labels'].append('[CLS]')
                ans['sentence'].append(buf)
                buf = []
                continue
            tmp = line.split()
            try:
                ans['toks'].append(tmp[0])
                ans['labels'].append(tmp[1])
                buf.append(tmp[0])
            except Exception as e:
                print("Exception at line no : %s"%lineIdx, e)

        if len(buf) == 0: # If the file is ending with a space : remove the last CLS
            ans['toks'] = ans['toks'][:-1] 
            ans['labels'] = ans['labels'][:-1]
        else: # If the file is not ending with a space
            ans['toks'].append('[SEP]')
            ans['labels'].append('[SEP]')
            ans['sentence'].append(buf)
            #count += len(buf)
            buf = []

    if debug:
        ansCount = 0
        prdCount = 0 
        with open(output_dir+'/NER_result_sent-debug.txt', 'w') as out_:
            for ans_sent, pred_sent in zip(ans['sentence'], bert_pred['sentence']):
                out_.write("ANS (%s): "%(len(ans_sent)) + " ".join(ans_sent)+"\n")
                out_.write("PRD (%s): "%(len(pred_sent)) + " ".join(pred_sent)+"\n")
                out_.write("\n")
                ansCount += len(ans_sent)
                prdCount += len(pred_sent)
                assert ansCount == prdCount, "\nans_sent : %s\npred_sent : %s"%(ans_sent, pred_sent)
    
        assert len(ans['sentence']) == len(bert_pred['sentence']), ( "len(ans['sentence'])(%s) == len(bert_pred['sentence'])(%s)"%(len(ans['sentence']), len(bert_pred['sentence'])))

        assert (len(ans['labels']) == len(bert_pred['labels'])), ("Error! : len(ans['labels'])(%s) != len(bert_pred['labels'])(%s) : Please report us"%(len(ans['labels']), len(bert_pred['labels'])))
    
    print("len(bert_pred['toks']): ", len(bert_pred['toks']), "len(ans['labels']): ", len(ans['labels']))
    with open(output_dir+'/NER_result_conll.txt', 'w') as out_:
        offset = 0 # Since some sentences can be trimmed due to max_seq_length, we use offset method.

        for idx, (bpred_t, bpred_l) in enumerate(zip(bert_pred['toks'], bert_pred['labels'])):
            if bpred_t=='[SEP]':
                if ans['labels'][idx+offset] != '[SEP]':
                    # When a sentence is trimmed ; len > max_seq_length -> find begining of new sentence.
                    print("## The predicted sentence of BioBERT model looks like trimmed. (The Length of the tokenized input sequence is longer than max_seq_length); Filling O label instead.")
                    print("   -> Showing 10 words near skipped part : %s"%" ".join(ans['toks'][idx+offset:idx+offset+11]))
                    for offIdx, label in enumerate(ans['labels'][idx+offset:]):
                        if label == '[SEP]':
                            offset += offIdx
                            break
                        else:
                            out_.write("%s %s-MISC %s-MISC\n"%(ans['toks'][idx+offset+offIdx], ans['labels'][idx+offset+offIdx], "O"))
                out_.write("\n")
            elif bpred_t=='[CLS]':
                pass
            else:
                try:
                    out_.write("%s %s-MISC %s-MISC\n"%(bpred_t, ans['labels'][idx+offset], bpred_l))
                except:
                    print("idx: ", idx, "offset: ", offset)

if __name__ == "__main__":
    bert_pred = detokenize(pred_token_test_path=args.token_test_path, pred_label_test_path=args.label_test_path)
    if args.answer_path is not None:
        transform2CoNLLForm(golden_path=args.answer_path, output_dir=args.output_dir, bert_pred=bert_pred, debug=args.debug)
    else:
        print("Golden answer not presented! Please use detokenize function.")
        raise NotImplementedError()
    print("NER detokenize done")
