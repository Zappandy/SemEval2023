# !/bin/sh

#source ../../semantic_snake/bin/activate  path to virtual env.

python train.py -lan es -m CenIA/albert-tiny-spanish -s tags -crf False -ss 1000
python eval.py -lan es -m CenIA/albert-tiny-spanish -s tags -crf False -ss 1000
