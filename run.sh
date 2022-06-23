export PYTHONPATH=.

#mkdir data
#cp orgs/train.csv orgs/dev.csv orgs/add.7z data/
#7z x data/add.7z
#mv add.csv data/add.csv

#cp data/train.csv data/train_dev.csv
#tail -n +2 data/dev.csv >> data/train_dev.csv

#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/make_folds.py --config configs/nfold.yaml

#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_rubert-base-cased/fold-0.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_rubert-base-cased/fold-1.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_rubert-base-cased/fold-2.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_rubert-base-cased/fold-3.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_rubert-base-cased/fold-4.yaml

#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/lower_distilrubert-tiny-cased-conversational-v1/fold-0.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/lower_distilrubert-tiny-cased-conversational-v1/fold-1.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/lower_distilrubert-tiny-cased-conversational-v1/fold-2.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/lower_distilrubert-tiny-cased-conversational-v1/fold-3.yaml
#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/lower_distilrubert-tiny-cased-conversational-v1/fold-4.yaml

#/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/distill.py --data_dir data_lower --input_file add.csv --preds_dir output/lower_lower_distilrubert-tiny-cased-conversational-v1

#cp data_lower/train.csv data_lower/full.csv
#tail -n +2 output/lower_lower_distilrubert-tiny-cased-conversational-v1/add.csv >> data_lower/full.csv

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/lower_distilrubert-tiny-cased-conversational-v1/final.yaml
