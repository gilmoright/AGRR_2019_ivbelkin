export PYTHONPATH=.

mkdir data
cp orgs/train.csv orgs/dev.csv orgs/add.7z data/
7z x data/add.7z
mv add.csv data/add.csv

cp data/train.csv data/train_dev.csv
tail -n +2 data/dev.csv >> data/train_dev.csv

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/make_folds.py --config configs/nfold.yaml

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/fold-0.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/fold-1.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/fold-2.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/fold-3.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/fold-4.yaml

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/fold-0.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/fold-1.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/fold-2.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/fold-3.yaml
/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/infer.py --input_file ../add.csv --checkpoint all --config configs/fold-4.yaml

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/distill.py --data_dir data --input_file add.csv --preds_dir output

cp data/train.csv data/full.csv
tail -n +2 output/add.csv >> data/full.csv

/s/ls4/users/grartem/anaconda3/envs/simptr/bin/python agrr/scripts/train.py --config configs/final.yaml
