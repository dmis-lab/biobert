# BioBERT
This repository provides the code for fine-tuning BioBERT, a biomedical language representation model designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc.
Please refer to our paper [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](http://doi.org/10.1093/bioinformatics/btz682) for more details.
This project is done by [DMIS-Lab](https://dmis.korea.ac.kr).

## Installation
Sections below describe the installation and the fine-tuning process of BioBERT based on Tensorflow.
If you are not familiar with coding and just want to recognize biomedical entities in your text using BioBERT, please use [this tool](https://bern.korea.ac.kr) which uses BioBERT for multi-type NER and normalization.

To fine-tune BioBERT, you need to download the [pre-trained weights of BioBERT](https://github.com/naver/biobert-pretrained).
After downloading the pre-trained weights, use `requirements.txt` to install BioBERT as follows:
```bash
$ git clone https://github.com/dmis-lab/biobert.git
$ cd biobert; pip install -r requirements.txt
```
Note that this repository is based on the [BERT repository](https://github.com/google-research/bert) by Google.
All the fine-tuning experiments were conducted on a single TITAN Xp GPU machine which has 12GB of RAM.
You might want to install `java` to use the official evaluation script of BioASQ. See `requirements.txt` for other details.

## Quick Links
Link | Detail
------------- | -------------
[Pre-trained weights](https://github.com/naver/biobert-pretrained) | Repository for pre-trained weights of BioBERT
[BERN](https://bern.korea.ac.kr) | Web-based biomedical NER + normalization using BioBERT
[7th BioASQ](https://github.com/dmis-lab/bioasq-biobert) | Code for the seventh BioASQ challenge winning model (factoid/yesno/list)
[Paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz682/5566506) | Paper link with [BibTeX](https://github.com/dmis-lab/biobert#citation) (Bioinformatics)

## FAQs
*   [How can I use BioBERT with PyTorch?](https://github.com/dmis-lab/biobert/issues/2)
*   [Can I get word/sentence embeddings using BioBERT?](https://github.com/dmis-lab/biobert/issues/23)
*   [How can I pre-train QA models on SQuAD?](https://github.com/dmis-lab/biobert/issues/10)
*   [What vocabulary does BioBERT use?](https://github.com/naver/biobert-pretrained/issues/1)

## Datasets
We provide a pre-processed version of benchmark datasets for each task as follows:
*   **[`Named Entity Recognition`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**: (17.3 MB), 8 datasets on biomedical named entity recognition
*   **[`Relation Extraction`](https://drive.google.com/open?id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw)**: (2.5 MB), 2 datasets on biomedical relation extraction
*   **[`Question Answering`](https://drive.google.com/open?id=19ft5q44W4SuptJgTwR84xZjsHg1jvjSZ)**: (5.23 MB), 3 datasets on biomedical question answering task.

You can simply run `download.sh` to download all the datasets at once.
```bash
$ ./download.sh
```
This will download the datasets under the folder `datasets`.
Due to the copyright issue of other datasets, we provide links of those datasets instead: **[`2010 i2b2/VA`](https://www.i2b2.org/NLP/DataSets/Main.php)**, **[`ChemProt`](http://www.biocreative.org/)**.

## Fine-tuning BioBERT
After downloading one of the pre-trained weights, unpack it to any directory you want, and we will denote this as `$BIOBERT_DIR`.
For instance, when using BioBERT-Base v1.1 (+ PubMed 1M), set `BIOBERT_DIR` environment variable as:
```bash
$ export BIOBERT_DIR=./biobert_v1.1_pubmed
$ echo $BIOBERT_DIR
>>> ./biobert_v1.1_pubmed
```

### Named Entity Recognition (NER)
Let `$NER_DIR` indicate a folder for a single NER dataset which contains `train_dev.tsv`, `train.tsv`, `devel.tsv` and `test.tsv`. Also, set `$OUTPUT_DIR` as a directory for NER outputs (trained models, test predictions, etc). For example, when fine-tuning on the NCBI disease corpus,
```bash
$ export NER_DIR=./datasets/NER/NCBI-disease
$ export OUTPUT_DIR=./ner_outputs
```
Following command runs fine-tuining code on NER with default arguments.
```bash
$ mkdir -p $OUTPUT_DIR
$ python run_ner.py --do_train=true --do_eval=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=10.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating `test.tsv`.
The token-level evaluation result will be printed as stdout format.
For example, the result for NCBI-disease dataset will be like this:
```
INFO:tensorflow:***** token-level evaluation results *****
INFO:tensorflow:  eval_f = 0.8972311
INFO:tensorflow:  eval_precision = 0.88150835
INFO:tensorflow:  eval_recall = 0.9136615
INFO:tensorflow:  global_step = 2571
INFO:tensorflow:  loss = 28.247158
```
(tips : You should go up a few lines to find the result. It comes before `INFO:tensorflow:**** Trainable Variables ****` )

Note that this result is the token-level evaluation measure while the official evaluation should use the entity-level evaluation measure. 
The results of `python run_ner.py` will be recorded as two files: `token_test.txt` and `label_test.txt` in `$OUTPUT_DIR`.
Use `./biocodes/ner_detokenize.py` to obtain word level prediction file.
```bash
$ python biocodes/ner_detokenize.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR
```
This will generate `NER_result_conll.txt` in `$OUTPUT_DIR`.
Use `./biocodes/conlleval.pl` for entity-level exact match evaluation results.
```bash
$ perl biocodes/conlleval.pl < $OUTPUT_DIR/NER_result_conll.txt
```

The entity-level results for the NCBI disease corpus will be like:
```
processed 24497 tokens with 960 phrases; found: 983 phrases; correct: 852.
accuracy:  98.49%; precision:  86.67%; recall:  88.75%; FB1:  87.70
             MISC: precision:  86.67%; recall:  88.75%; FB1:  87.70  983
``` 
Note that this is a sample run of an NER model.
The performance of NER models usually converges at more than 50 epochs (learning rate = 1e-5 is recommended).

### Relation Extraction (RE)
Let `$RE_DIR` indicate a folder for a single RE dataset, `$TASK_NAME` denote the name of task (two possible options: {gad, euadr}), and `$OUTPUT_DIR` denote a directory for RE outputs:
```bash
$ export RE_DIR=./datasets/RE/GAD/1
$ export TASK_NAME=gad
$ export OUTPUT_DIR=./re_outputs_1
```
Following command runs fine-tuining code on RE with default arguments.
```bash
$ python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
```
The predictions will be saved into a file called `test_results.tsv` in the `$OUTPUT_DIR`.
Use `./biocodes/re_eval.py` for the evaluation.
Note that the CHEMPROT dataset is a multi-class classification dataset and to evaluate the CHEMPROT result, you should run `re_eval.py` with additional `--task=chemprot` flag.
```bash
$ python ./biocodes/re_eval.py --output_path=$OUTPUT_DIR/test_results.tsv --answer_path=$RE_DIR/test.tsv
```
The result for GAD dataset will be like this:
```
f1 score    : 83.74%
recall      : 90.75%
precision   : 77.74%
specificity : 71.15%
```
Please be aware that you have to change `$OUTPUT_DIR` to train/test a new model.
For instance, as most RE datasets are in 10-fold, you have to make a different output directory to train/test a model for a different fold (e.g., `$ export OUTPUT_DIR=./re_outputs_2`).

### Question Answering (QA)
To use the BioASQ dataset, you need to register in the [BioASQ website](http://participants-area.bioasq.org/general_information/general_information_registration/) which authorizes the use of the dataset.
Please unpack the pre-processed BioASQ dataset provided above to a directory `$QA_DIR`.
For example, with `$OUTPUT_DIR` for QA outputs, set as:
```bash
$ export QA_DIR=./datasets/QA/BioASQ
$ export OUTPUT_DIR=./qa_outputs
```
Files named as `BioASQ-*.json` are used for training and testing the model which are the pre-processed format for BioBERT.
Note that we pre-trained our model on SQuAD dataset to get state-of-the-art performance (see [here](https://github.com/dmis-lab/bioasq-biobert) to get BioBERT pre-trained on SQuAD), and you might have to change `$BIOBERT_DIR` accordingly.
Following command runs fine-tuining code on QA with default arguments.
```bash
$ python run_qa.py --do_train=True --do_predict=True --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --max_seq_length=384 --train_batch_size=12 --learning_rate=5e-6 --doc_stride=128 --num_train_epochs=5.0 --do_lower_case=False --train_file=$QA_DIR/BioASQ-train-factoid-4b.json --predict_file=$QA_DIR/BioASQ-test-factoid-4b-1.json --output_dir=$OUTPUT_DIR
```
The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in `$OUTPUT_DIR`.
Run `./biocodes/transform_nbset2bioasqform.py` to convert `nbest_predictions.json` to the BioASQ JSON format, which will be used for the official evaluation.
```bash
$ python ./biocodes/transform_nbset2bioasqform.py --nbest_path=$OUTPUT_DIR/nbest_predictions.json --output_path=$OUTPUT_DIR
```
This will generate `BioASQform_BioASQ-answer.json` in `$OUTPUT_DIR`.
Clone **[`evaluation code`](https://github.com/BioASQ/Evaluation-Measures)** from BioASQ github and run evaluation code on `Evaluation-Measures` directory. Please note that you should always put 5 as parameter for -e.
```bash
$ git clone https://github.com/BioASQ/Evaluation-Measures.git
$ cd Evaluation-Measures
$ java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ../$QA_DIR/4B1_golden.json ../$OUTPUT_DIR/BioASQform_BioASQ-answer.json
```
As our model is only on factoid questions, the result will be like,
```
0.0 0.3076923076923077 0.5384615384615384 0.394017094017094 0.0 0.0 0.0 0.0 0.0 0.0
```
where the second, third and fourth numbers will be SAcc, LAcc, and MRR of factoid questions respectively.
For list and yes/no type questions, please refer to our repository for [BioBERT at the 7th BioASQ Challenge](https://github.com/dmis-lab/bioasq-biobert).

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.

## Citation
```
@article{10.1093/bioinformatics/btz682,
    author = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
    title = "{BioBERT: a pre-trained biomedical language representation model for biomedical text mining}",
    journal = {Bioinformatics},
    year = {2019},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz682},
    url = {https://doi.org/10.1093/bioinformatics/btz682},
}
```

## Contact Information
For help or issues using BioBERT, please submit a GitHub issue. Please contact Jinhyuk Lee
(`lee.jnhk (at) gmail.com`), or Wonjin Yoon (`wonjin.info (at) gmail.com`) for communication related to BioBERT.
