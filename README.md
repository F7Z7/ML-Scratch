# Neural Network from Scratch (NumPy)

A fundamental two-hidden-layer neural network built from scratch using NumPy — updated to reflect the repository's current structure.

---

## Overview

This repository implements a basic feed-forward neural network with two hidden layers using only `NumPy`. It demonstrates core neural network components (Layers, Activations, Loss) and provides example scripts that train and evaluate models on small standard datasets (Breast Cancer, Iris, Wine).

---

## Why this exists

This project started as a learning exercise to build neural networks from the ground up and remains a didactic resource for anyone who wants to understand the low-level math and code behind neural networks without relying on high-level frameworks.

---

## 1. Core Neural Network Components

- Dense Layer — `layers/dense.py`  
  Implements a fully-connected (dense) layer with He initialization (`np.sqrt(2 / inputs)`), forward pass (`y = xW + b`), backward pass (computes `dweights`, `dbiases`, `dinputs`), and `update_params()` to apply learning-rate updates.

- ReLU Activation — `activations/activation_relu.py`  
  `forward()` computes `np.maximum(0, inputs)`; `backward()` zeros gradients where inputs ≤ 0.

- Softmax Activation — `activations/activation_softmax.py`  
  Numerically-stable softmax implementation (subtracts the row-wise max before exponentiation).

- Cross-Entropy Loss — `loss/cross_entropy.py`  
  Uses clipping (`np.clip(..., 1e-7, 1 - 1e-7)`) for numerical stability and contains a backward that works with Softmax (combined derivative when applicable).

---

## 2. Helper scripts & utilities

- `datasets.py` — helper for loading and preprocessing datasets (present in repo root).
- `methods/train_test.py` — demonstrates `train_test_split()` usage.
- `methods/onehotencod.py` — demonstrates one-hot encoding of labels.

---

## 3. Model templates, configs & example scripts

Each example follows a common 3-layer pattern:

Input Layer (Dense + ReLU) → Hidden Layer (Dense + ReLU) → Output Layer (Dense + Softmax)

- `models/base_model.py` — template base model used as a starting point for dataset-specific scripts. It contains reusable architecture construction, training loop scaffolding, and common helper functions so new models can be added quickly by subclassing or importing this template.
- `models/*config*` — a configuration mechanism is used to centralize hyperparameters and training settings (learning rate, epochs, layer sizes, batch size, etc.). 
- `main.py` — example/training script (Breast Cancer dataset).
- `models/breastCancer_Model.py` — Breast Cancer model script (example using the base template + config).
- `models/iris_test_model.py` — Iris model script.
- `models/wineDataset_test_model.py` — Wine dataset script.


---

## Repository structure (current)

├── activations/  
│   ├── activation_relu.py  
│   └── activation_softmax.py  
├── layers/  
│   └── dense.py  
├── loss/  
│   └── cross_entropy.py  
├── methods/  
│   ├── train_test.py  
│   └── onehotencod.py  
├── models/  
|── |── config.py ← for setting up the parameters of each model
│   ├── base_model.py        ← template base model (reusable)  
│   ├── breastCancer_Model.py  
│   ├── iris_test_model.py  
│   └── wineDataset_test_model.py  
├── images/  
│   ├── breast_results.png  
│   ├── wine_results.jpg  
│   └── iris_results.jpg  
├── datasets.py  
├── main.py  
└── requirements.txt

---

## Quick usage

1. Create a virtual environment and install dependencies:

   pip install -r requirements.txt

2. Run an example (e.g., Breast Cancer):

   python main.py

3. To create a new model:
   - Copy or import `models/base_model.py`.
   - Add a config entry (or new config file) specifying layer sizes, LR, epochs, etc.
   - Implement dataset-specific preprocessing and call the base training loop.

4. Inspect the other example scripts in `models/` and the helper scripts in `methods/`.

---

## Dependencies

- numpy  //core library
- matplotlib  // for plotting 
- scikit-learn  //for datasets and confusion matrices
- nnfs (optional; used in some helper/testing code) //initially used for simple datasets

These are listed in `requirements.txt`.

---

## Results

The example models provide strong baselines and have previously achieved ~90%+ accuracy on some of the small datasets (Breast Cancer, Wine, Iris). See the `images/` directory for saved result plots.

---

## Contributing

Contributions, issues, and feature requests are welcome. If you add a new model or script, please update this README to reflect the new structure and usage notes.

---

## License & attribution

This project is intended for learning and demonstration. If you reuse substantial parts of the code or documentation, please attribute the original author.

---
