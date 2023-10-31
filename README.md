# Federated Learning Uncertainty Estimation: Investigating Factors for Improved Quality

## Weekly Meeting Feedback
### 2023/10/30
To-do list:
- Visualization:
  - Trending Figures, according to the Appendix of H. Brendan McMahan's paper.
  - Heatmaps, to show the IID and Non-IID distribution strategy.
  
- Dataset 
Validate the correctness of Fed-Avg and then try to apply to these datasets:
  - Raman spectrum, e.g., the datasets from http://mikkelschmidt.dk/papers/li2022analyst.pdf
  - SERS maps, e.g., the datasets from http://mikkelschmidt.dk/papers/li2023analyst.pdf (funding)

- week plan
listed as below.

- HPC
  - try some toy projects.
  - https://docs.google.com/document/d/1pBBmoLTj_JPWiCSFYzfHj646bb8uUCh8lMetJxnE68c/edit
  - https://www.hpc.dtu.dk/?page_id=2534

## Milestones

- w8(10/23-10/29) **(3/3 Finished)**
  - Implemented the iid and non-iid data assignment function to simulate the two corresponding scenarios in the paper.
  - Secondly we implemented the Fed-Avg algorithm in H. Brendan McMahan's paper and tried to reproduce the published results on the MNIST dataset.
  - Tested different C value(0.1, 0.2, 0.5) in both iid and non-iid in both 2NN and CNN model.

- w9(10/30 - 11/05):
    - improve the current Fed-Avg:
      - Implement Visualization module.(Trending Figures, Heatmaps)
      - Debug the CNN model in Fed-Avg.
      - Implement the baseline (C = 0) and FedSGD (C = 1, B = inf) groups of test to get the comparison with current groups.
    - Read two papers about uncertainty estimation.

- w10(11/06 - 11/12):
    - Implement Uncertainty estimation Algorithm, integrate it into existing Fed-Avg framework, and validate it in MNIST dataset.
    - Apply Fed-Avg to SERS maps(or Raman spectrum). (1/2) 


- w11(11/13 - 11/19):
    - Apply Fed-Avg to SERS maps(or Raman spectrum). (2/2)

- w12(11/20 - 11/26):
    - Final report. (1/2)

- w13(11/27 - 12/03):
    - Final report. (2/2) 

## References:
- Federated learning: https://arxiv.org/pdf/1602.05629.pdf

- Uncertainty estimation: 
  - https://arxiv.org/pdf/1612.01474.pdf
  - https://arxiv.org/pdf/1703.02910.pdf 
