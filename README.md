# Federated Learning Uncertainty Estimation: Investigating Factors for Improved Quality

## Weekly Meeting Feedback
### 2023/11/19
To-do list:
- Visualization:
  - About data distribution: heatmap of data distribution of IID and non-IID.
  - About accuracy: Compare overall accuracy VS round for different parameters.
  - About uncertainty: 
    - Uncertainty (range and mean) VS rounds in every group.
    - Variance of uncertainty (Variation Ratio) v.s. Accuracy. (the third figure below.) (should we abort this one?)
  - About calibaration
    - Predicted Probability VS Accuracy 
    - Entrophy VS Accuracy (?)
    - ECE VS Rounds

- About Entrophy:
  - implement another uncertainty calculate functions, such as BALD.

- Datasets:
  - Raman spectrum, e.g., the datasets from http://mikkelschmidt.dk/papers/li2022analyst.pdf
  - SERS maps, e.g., the datasets from http://mikkelschmidt.dk/papers/li2023analyst.pdf (funding)

- HPC
  Done.

## Milestones

- w8(10/23-10/29) **(3/3 Finished)**
  - Implemented the iid and non-iid data assignment function to simulate the two corresponding scenarios in the paper.
  - Secondly we implemented the Fed-Avg algorithm in H. Brendan McMahan's paper and tried to reproduce the published results on the MNIST dataset.
  - Tested different C value(0.1, 0.2, 0.5) in both iid and non-iid in both 2NN and CNN model.

- w9(10/30 - 11/05): **(4/4 Finished)**
    - improve the current Fed-Avg:
      - Implement Visualization module.(Trending Figures, Heatmaps) **(11/1 Finished)**
      - Debug the CNN model in Fed-Avg. **(11/2 Finished)**
      - Implement the baseline (C = 0) and FedSGD (C = 1, B = inf) groups of test to get the comparison with current groups. **(11/5 basicly finished, cannot implement baseline groups in my PC. HPC needed.)**
    - Read two papers about uncertainty estimation. **(11/5 Finished)**

- w10(11/06 - 11/12):**(1/1 Finished)**
    - Implement Uncertainty estimation Algorithm, integrate it into existing Fed-Avg framework, and validate it in MNIST dataset. **(11/5 Finished)** 


- w11(11/13 - 11/19):**(2/3 Finished)**
    - Modified for HPC. Completed script for parallel computing in DCC. **(11/15 Finished)**
    - Implemented Expected Calibration Error(ECE) for calibration evaluation.**(11/17 Finished)**
    - Apply Fed-Avg to SERS maps(or Raman spectrum). (?)

- w12(11/20 - 11/26):
    - Final report. (1/2)

- w13(11/27 - 12/03):
    - Final report. (2/2) 

## References:
- Federated learning: https://arxiv.org/pdf/1602.05629.pdf

- Uncertainty estimation: 
  - https://arxiv.org/pdf/1612.01474.pdf
  - https://arxiv.org/pdf/1703.02910.pdf 
