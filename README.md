# Naive Bayes Classifier

计95 刘玉河 2019011560 liuyuhe19@mails.tsinghua.edu.cn

## File Structure
```text
src
├── configs                         # Config Files
│   ├── baseline.yaml
│   ├── content.yaml
│   ├── tf_idf_content.yaml
│   └── tf-idf.yaml
├── dataloader.py                   # Dataloader
├── dataset                         # Dataset
│   ├── data
│   └── label
├── eval.py                         # Performance Evaluation
├── main.py                 # Main Program
├── match.py                        # Specific Features
├── model.py                        # NBC model
├── params.py                       # Params
├── README.md
├── report.pdf
├── requirements.txt
├── test.py                         # Testing Pipeline
└── workdir
```
## How to run the code

1. Install required python packages in `requirements.txt`
2. Copy the dataset to folder `dataset` to make sure it has two sub-folders including `data` and `label`
3. run `python main.py [--config <config_file>] [<params>]`

Note: Random seed is 1 by default, you don't have to change it to replay the result in the report.

(Examples below)

```text
Naive Bayes Classification

optional arguments:
  -h, --help            show this help message and exit
  --config              specify the config yaml file
  --data_path           specify data_path
  --label_path          specify label_path
  --seed                1 by default
  --k_fold              5 by default
  --save_result         save filename
  --sample              sample_rate
  --alpha               ALPHA in zero-probablity
  --word_model          {bow,tf-idf}
  --n_gram              N_GRAM       
  --vocabulary_volume   VOCABULARY_VOLUME
  --content_feature     {True, False}， whether to extract specific features
```

## Example

```bash
python main.py --config ./configs/baseline.yaml
```
Run the baseline model, see the result in `src/workdir/result_baseline.txt`

```
# ./workdir/result_baseline.txt

# Accuracy Precision Recall F1
0.753 0.9752 0.6407 0.7734      # fold 0 result
0.7567 0.9762 0.648 0.779       # fold 1
0.7492 0.9817 0.6323 0.7692
0.748 0.9801 0.6349 0.7706
0.7639 0.9779 0.6498 0.7808     # fold 4
0.7542 0.9782 0.6411 0.7746     # Macro 
0.7542 0.9782 0.6411 0.7746     # Micro
```

more config files are in `src/configs`

```text
baseline.yaml           BOW with no extra features
content.yaml            BOW with extra features
tf-idf.yaml             TF-IDF with no extra features
tf_idf_content.yaml     TF-IDF with extra features
```