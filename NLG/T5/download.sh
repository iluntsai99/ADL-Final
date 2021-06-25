#!bin/bash

mkdir -p ./ckpt/model
wget https://www.dropbox.com/s/5zzpiabijhi7r4k/config.json?dl=1 -O ./ckpt/model/config.json
wget https://www.dropbox.com/s/e1436x4vmjrr5rj/pytorch_model.bin?dl=1 -O ./ckpt/model/pytorch_model.bin