# Relation-Extraction
Project of Classification of Semantic Relations

Require:
- Python 3.6
- Keras 2.2.0 and TensorFlow (1.8.0) as backend
- Dependency-Based Word Embeddings from Omer Levy and Yoav Goldberg
- nltk treebank

## Usage
dataset are put under data/
### show help message
```
$ python [clean_data, preprocess, train_predict].py --help
```
### Preprocess
* please create directory pkl/ first
* example:
  ```
  $ python clean_data.py
  $ python preprocess.py
  ```
* You can load bin type embeddings with ```--bin```
### Train and Predict
* please create directory result/ first
* example:
  ```
  $ python train_predict.py 
  ```
* predict with pretrained model:
  ```
  $ python train_predict.py -l [result/CNNmodel.h5]
  ```
### Evaluate
```
$ perl semeval2010_task8_scorer-v1.2.pl result/predict.txt answer_key.txt
```
