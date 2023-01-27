# !/bin/sh

#source ../../semantic_snake/bin/activate  path to virtual env.

#python train.py -lan zh -m bert-base-chinese -s LM -crf False -ss 6000
python eval.py -lan zh -m bert-base-chinese -s LM -crf False -ss 6000

#python eval.py -lan zh -m xlm-roberta-large -s tags -crf False -ss 6000
