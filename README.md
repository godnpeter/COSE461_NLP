Korea Univ. COSE461 Natural Language Processing

## English
```
git clone https://github.com/mynsng/COSE461_NLP.git

cd COSE461_NLP/English

conda create -n Bert_Friends python=3.7

conda install pytorch=1.5.0 torchvision cuda100 cudatoolkit=9.2 -c pytorch

source activate Bert_Friends

python -m pip install -r requirements.txt

python Bert_Friends_test.py

```

## Korean
```
git clone https://github.com/mynsng/COSE461_NLP.git

cd COSE461_NLP/kobert

conda create -n kobert python=3.6

conda activate kobert

python -m pip install -r requirements.txt

if use linear scheduler

  python -u train.py --scheduler linear

elif use cosine scheduler

  python -u train.py --scheduler cosine

```
