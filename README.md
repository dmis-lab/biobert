# BioBERT
This repository provides fine-tuning codes of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. Please refer to our paper [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](http://arxiv.org/abs/1901.08746) for more details.

## Updates
*   **(3 Feb 2019)** Updated our [arxiv paper](http://arxiv.org/abs/1901.08746).

## Installation
To use BioBERT, we need pre-trained weights of BioBERT, which you can download from [Naver GitHub repository for BioBERT pre-trained weights](https://github.com/naver/biobert-pretrained). Make sure to specify the versions of pre-trained weights used in your works. Also, note that this repository is based on the [BERT repository](https://github.com/google-research/bert) by Google.

All the fine-tuning experiments were conducted on a single TITAN Xp GPU machine which has 12GB of RAM. The code was tested with Python2 and Python3 (We used Python2 for experiments). You might want to install `java` to use official evaluation script of BioASQ. See `requirements.txt` for other details.

## Datasets
We provide pre-processed version of benchmark datasets for each task as follows:
*   **[`Named Entity Recognition`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**: (17.3 MB), 8 datasets on biomedical named entity recognition
*   **[`Relation Extraction`](https://drive.google.com/open?id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw)**: (2.5 MB), 2 datasets on biomedical relation extraction
*   **[`Question  Answering`](https://drive.google.com/open?id=1R2aTOdvGlce95OVQLXnJ_pdm3i6UmVOw)**: (1.10 MB), 2 datasets on biomedical question answering task.

For details on NER datasets, please see **A Neural Network Multi-Task Learning Approach to Biomedical Named Entity Recognition (Crichton et al. 2017)**.
The source of pre-processed datasets are from https://github.com/cambridgeltl/MTL-Bioinformatics-2016 and https://github.com/spyysalo/s800.

For details on QA datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.

Due to the copyright issue of some datasets, we provide links of those datasets instead:
### NER
*   **[`2010 i2b2/VA`](https://www.i2b2.org/NLP/DataSets/Main.php)**

### RE
*   **[`ChemProt`](http://www.biocreative.org/)**

### QA
*   **[`BioASQ Task B`](http://participants-area.bioasq.org/Tasks/A/getData/)**

## Fine-tuning BioBERT
After downloading one of the pre-trained models from [Naver GitHub repository for BioBERT pre-trained weights](https://github.com/naver/biobert-pretrained), unpack it to any directory you want, which we will denote as `$BIOBERT_DIR`. 

### Named Entity Recognition (NER)
Download and unpack the NER datasets provided above (**[`Named Entity Recognition`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**). From now on, `$NER_DIR` indicates a folder for a single dataset which should include `train_dev.tsv`, `train.tsv`, `devel.tsv` and `test.tsv`. For example, `export NER_DIR=~/bioBERT/biodatasets/NERdata/NCBI-disease`. Following command runs fine-tuining code on NER with default arguments.
```
mkdir /tmp/bioner/
python run_ner.py \
    --do_train=true \
    --do_eval=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$NER_DIR/ \
    --output_dir=/tmp/bioner/
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating `test.tsv`.
The token-level evaluation result will be printed as stdout format. For example, the result for NCBI-disease dataset will be like this:
```
INFO:tensorflow:***** token-level evaluation results *****
INFO:tensorflow:  eval_f = 0.9028707
INFO:tensorflow:  eval_precision = 0.8839457
INFO:tensorflow:  eval_recall = 0.92273223
INFO:tensorflow:  global_step = 2571
INFO:tensorflow:  loss = 25.894125
```
(tips : You should go up a few lines to find the result. It comes before `INFO:tensorflow:**** Trainable Variables ****` )

Note that this result is the token-level evaluation measure while the official evaluation should use the entity-level evaluation measure. 
The results of `python run_ner.py` will be recorded as two files: `token_test.txt` and `label_test.txt` in `output_dir`. 
Use `ner_detokenize.py` in `./biocodes/` to obtain word level prediction file.
```
python biocodes/ner_detokenize.py \
--token_test_path=/tmp/bioner/token_test.txt \
--label_test_path=/tmp/bioner/label_test.txt \
--answer_path=$NER_DIR/test.tsv \
--output_dir=/tmp/bioner
```
This will generate `NER_result_conll.txt` in `output_dir`.
Use `conlleval.pl` in `./biocodes/` for entity-level exact match evaluation results.
```
perl biocodes/conlleval.pl < /tmp/bioner/NER_result_conll.txt
```

The entity-level results for NCBI-disease dataset will be like :
```
processed 24497 tokens with 960 phrases; found: 993 phrases; correct: 866.
accuracy:  98.57%; precision:  87.21%; recall:  90.21%; FB1:  88.68
             MISC: precision:  87.21%; recall:  90.21%; FB1:  88.68  993
``` 
Note that this is a sample run of an NER model. Performance of NER models usually converges at more than 50 epochs (learning rate = 1e-5 is recommended).

### Relation Extraction (RE)
Download and unpack the RE datasets provided above (**[`Relation Extraction`](https://drive.google.com/open?id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw)**). From now on, `$RE_DIR` indicates a folder for a single dataset. `{TASKNAME}` means the name of task such as gad or euadr. For example, `export RE_DIR=~/bioBERT/biodatasets/REdata/GAD/1` and `--task_name=gad`. Following command runs fine-tuining code on RE with default arguments.
```
python run_re.py \
    --task_name={TASKNAME} \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --do_lower_case=false \
    --data_dir=$RE_DIR/ \
    --output_dir=/tmp/RE_output/ 
```
The predictions will be saved into a file called `test_results.tsv` in the `output_dir`. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating test.tsv. Use `./biocodes/re_eval.py` in `./biocodes/` folder for evaluation. Also, note that CHEMPROT dataset is a multi-class classification dataset. To evaluate CHEMPROT result, run `re_eval.py` with additional `--task=chemprot` flag.
```
python ./biocodes/re_eval.py --output_path={output_dir}/test_results.tsv --answer_path=$RE_DIR/test.tsv
```
The result for GAD dataset will be like this:
```
.tsv
recall      : 92.88%
specificity : 67.19%
f1 score    : 83.52%
precision   : 75.87%
```
Please be aware that you have to move `output_dir` to make new model. As some RE datasets are 10-fold divided, you have to make different output directories to train a model with different datasets.

### Question Answering (QA)
To download QA datasets, you should register in [BioASQ website](http://participants-area.bioasq.org). After the registration, download **[`BioASQ Task B`](http://participants-area.bioasq.org/Tasks/A/getData/)** data, and unpack it to some directory `$BIOASQ_DIR`. Finally, download **[`Question Answering`](https://drive.google.com/open?id=1R2aTOdvGlce95OVQLXnJ_pdm3i6UmVOw)**, our pre-processed version of BioASQ-4/5b datasets, and unpack it to `$BIOASQ_DIR`.

Please use `BioASQ-*.json` for training and testing the model. This is necessary as the input data format of BioBERT is different from BioASQ dataset format. Also, please be informed that the do_lower_case flag should be set as `--do_lower_case=False`. Following command runs fine-tuining code on QA with default arguments.
```
python run_qa.py \
     --do_train=True \
     --do_predict=True \
     --vocab_file=$BIOBERT_DIR/vocab.txt \
     --bert_config_file=$BIOBERT_DIR/bert_config.json \
     --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt \
     --max_seq_length=384 \
     --train_batch_size=12 \
     --learning_rate=3e-5 \
     --doc_stride=128 \
     --num_train_epochs=50.0 \
     --do_lower_case=False \
     --train_file=$BIOASQ_DIR/BioASQ-train-4b.json \
     --predict_file=$BIOASQ_DIR/BioASQ-test-4b-1.json \
     --output_dir=/tmp/QA_output/
```
The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run `transform_nbset2bioasqform.py` in `./biocodes/` folder to convert `nbest_predictions.json` to BioASQ JSON format, which will be used for the official evaluation.
```
python ./biocodes/transform_nbset2bioasqform.py --nbest_path={QA_output_dir}/nbest_predictions.json --output_path={output_dir}
```
This will generate `BioASQform_BioASQ-answer.json` in `{output_dir}`.
Clone **[`evaluation code`](https://github.com/BioASQ/Evaluation-Measures)** from BioASQ github and run evaluation code on `Evaluation-Measures` directory. Please note that you should always put 5 as parameter for -e.
```
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $BIOASQ_DIR/4B1_golden.json \
    RESULTS_PATH/BioASQform_BioASQ-answer.json
```
As our model is only on factoid questions, the result will be like
```
0.0 0.4358974358974359 0.6153846153846154 0.5072649572649572 0.0 0.0 0.0 0.0 0.0 0.0
```
where the second, third and fourth numbers will be SAcc, LAcc and MRR of factoid questions respectively.
Note that we pre-trained our model on SQuAD dataset to get the state-of-the-art performance. Please check our paper for details.

## License and Disclaimer
Please see LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.

## Citation

For now, cite [the Arxiv paper](http://arxiv.org/abs/1901.08746):

```
@article{lee2019biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={arXiv preprint arXiv:1901.08746},
  year={2019}
}
```

If we submit the paper to a conference or journal, we will update the BibTeX.

## Contact information

For help or issues using BioBERT, please submit a GitHub issue. Please contact Jinhyuk Lee
(`lee.jnhk (at) gmail.com`), or Wonjin Yoon (`wonjin.info (at) gmail.com`) for communication related to BioBERT.
