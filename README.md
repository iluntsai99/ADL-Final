# README for ADL-Final NTU 109 Spring
## NLG

***Note that the datas have already been preprocessed and saved in folder "dataset". If you want to preprocess yourself, please checkout folder "preprocess"**

### Download

```shell
git clone https://github.com/iluntsai99/ADL-Final
# download models
cd ADL-Final/NLG/T5/
bash download.sh
```

### Training

```shell
# training
python train.py
```

### Inference

```shell
# testing
python inference.py
# chit-chat result in result.json
```

### Filtering
```shell
# filtering
LEO PLZ
```



## DST

**Note that the dataset has already been preprocessed and saved in zipped folder "data"**

### Download, extract data, and dependencies

```shell
git clone https://github.com/iluntsai99/ADL-Final
# download models
cd ADL-Final/DST/
bash download_glove.sh
# Unzip preprocessed data (original data can be used if only testing is involved)
unzip data.zip
# Install dependencies
pip install -r requirements.txt
```

### Training

```shell
# Preprocess
python train_process.py
python dev_generate_answer.py
python dev_to_test_format.py

# training
bash train.sh

# validation
bash dev.sh
python final_answer_process.py dev
```

### Inference

```shell
# testing
bash test_seen.sh
bash test_unseen.sh
# csv submission result is seen_submission.csv and unseen_submission.csv respectively
```
**Note: For postprocessing with glove, please switch to use final_answer_process_with_glove.py instead of final_answer_process.py**

### Model Link

drive: https://drive.google.com/file/d/1usvvvzcqwdaOO7yKUlVyt8VQYep2ctjg/view?usp=sharing


### 講稿&投影片

drive: https://drive.google.com/drive/folders/1DpuoM_X41M1oFr02QizUG-rBIRytgvuV?usp=sharing
