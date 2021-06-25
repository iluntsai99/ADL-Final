#!bin/bash

mkdir -p ./ckpt/model
wget https://www.dropbox.com/s/g872mdsbjs6rplf/config.json?dl=1 -O ./ckpt/model/config.json
wget https://www.dropbox.com/s/umofen6d4huce50/pytorch_model.bin?dl=1 -O ./ckpt/model/pytorch_model.bin