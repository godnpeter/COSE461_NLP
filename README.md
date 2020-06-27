Korea Univ. COSE461 Natural Language Processing

## English
```

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
