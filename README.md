# BALAD
ICASSP24 - Time Series Anomaly Detection

This is the source code of the paper  **"BOUNDARY-DRIVEN ACTIVE LEARNING FOR ANOMALY DETECTION IN TIME SERIES DATA STREAMS"** accepted by ICASSP24 (to appear).

# "## Citation"

# "Please cite our paper if you find this code is useful.  "

# "Zhou Xiaohui, Wang Yijie, Xu Hongzuo, Liu Mingyu. BOUNDARY-DRIVEN ACTIVE LEARNING FOR ANOMALY DETECTION IN TIME SERIES DATA STREAMS"
## Usage
1. run main.py for sample usage.
2. Data set: You may want to find the sample input data set in the "datasets" folder.
2. Pre-trained models: You may want to utilize the pre-trained models in the "pretrain_models" folder.
3. The input path can be an individual data set or just a folder. 
4. The performance might have slight differences between two independent runs. In our paper, we report the average AUC-ROC, AUC-PR with std over 5 runs. 
5. sDSADS, sPSM and sSWaT are too large, and the download link will be available soon.

## Dependencies
```
Python 3.7
Troch == 1.13.1+cu116
pandas == 1.3.5
scikit-learn == 1.0.2
numpy == 1.21.6
```
