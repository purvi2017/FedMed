# FedMed

Federated learning for brain tumor segmentation on MRI scans.

Hospitals usually cannot share patient scans. In federated learning, each
hospital trains the model on its own data and only the model weights are
sent to a central server, which averages them. FedMed is a small project
that tries this idea on brain MRI.

The hospitals are simulated as separate clients on one machine.

## Status

Work in progress. Nothing has been trained yet.

## Planned work

- [x] Load and preprocess MRI scans
- [ ] Train a 3D U-Net baseline on one machine
- [ ] Federated training with Flower (3 clients, FedAvg)
- [ ] Privacy features (differential privacy, secure aggregation)
- [ ] Dashboard for training metrics

## Tech

Python 3.11, PyTorch, MONAI, Flower

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
