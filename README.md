# Homework 1 ADL NTU
> This is a homework for Applied Deep Learning

## Environment
```shell
# build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```
## Download
```shell
# To download required file
bash download.sh
```

## Intent detection
```shell
# To trian
python3 train_intent.py
# To test
bash ./slot_tag.sh ./data/slot/test.json pred_intent.csv
```
## Slot classification
```shell
# To train
python3 train_intent.py --num_eopch 25
# To test
bash ./intent_cls.sh ./data/intent/test.json pred_slot.csv
```
