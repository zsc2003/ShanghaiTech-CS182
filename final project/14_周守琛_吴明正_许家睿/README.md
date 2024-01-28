# ShanghaiTech-CS182-Final-Project
ShanghaiTech CS182 Introduction to Machine Learning final project, Fall 2023.


## Group ID:
$14$


## Group members:

name  student ID   email

周守琛 2021533042 zhoushch@shanghaitech.edu.cn

吴明正 2021533066 wumzh@shanghaitech.edu.cn

许家睿 2021533092 xujr@shanghaitech.edu.cn


## dataset

Since the dataset is about $25$GB, so it is hard to upload. So we put the data into the ShanghaiTech cloud disk:
https://epan.shanghaitech.edu.cn/l/sF3Afc 

The MSMS is the outer library for preprocessing the data.


## environment
The code wa run in python 3.7.3, and the required packages are listed in the file 'requirements.txt'.

CUDA Version: 11.4

GPU Memory: 24GB


## Run the code:

To run the code, open the terminal at current directory(same folder with this README.md file), and run the following command:

### 1. Preprocessing
run the code in the folder 'masif' sequentially in chronological order.

### 2. Training
```
python ./code/Pocketclassification/main.py --name pocket_classification --processed_dir <pre-computed data>
```

### 3. Inference
```
python ./code/Pocketclassification/predict.py --pred_out_dir ./data/inference
```


## results

The results can be seen in the folder: 'result_figures'.


## Contribution:

Code were written by all the group members with github collaboration. And the report was written by all the group members through overleaf collaboration. So the contribution is equal.