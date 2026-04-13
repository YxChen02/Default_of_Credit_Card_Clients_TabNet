# Credit Card Default Prediction with TabNet

This repository contains a Jupyter Notebook that demonstrates the use of **TabNet**, a deep learning model designed specifically for tabular data, to predict credit card defaults. The model is trained and evaluated on the [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) from the UCI Machine Learning Repository.

## Repository Contents

- `Default_of_Credit_Card_Clients_TabNet.ipynb` – Complete analysis and modeling pipeline:
  - Data loading via `ucimlrepo`
  - Train/validation/test splitting and standardization
  - TabNet classifier training with GPU acceleration
  - Evaluation using AUC metric
  - Feature importance and interpretability visualizations (mask heatmaps, global importance)

## Dataset

- **Source**: [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Samples**: 30,000
- **Features**: 23 (including demographic factors, credit history, payment status, and bill amounts)
- **Target**: Binary – default payment next month (Yes = 1, No = 0)

## Model Performance
- Best Validation AUC: ~0.770 (achieved at epoch 47)
- Early Stopping: Patience of 20 epochs
- Optimizer: Adam with learning rate 0.02
- Scheduler: StepLR (step size 10, gamma 0.9)

## Interpretability
TabNet provides built‑in interpretability through sequential attention masks. The notebook includes:
- Step‑wise Mask Heatmaps – showing which features the model focuses on at each decision step for the first 50 test samples.
- Global Feature Importance – a horizontal bar chart ranking the 23 features by their average importance across all steps.
These visualizations help understand which factors (e.g., payment history, bill amounts) most influence the model's predictions.

## Reference
- Arik, S. O., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv:1908.07442
