# Decoding musical preference from EEG

## Overview

This project investigates whether binary musical preference (like versus dislike) can be decoded from single-trial EEG recordings. We use the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), which includes data from 32 subjects, each with 32-channel EEG recorded while listening to 40 one-minute music video excerpts. Our analyses explore a range of feature types, including differential entropy, frontal alpha asymmetry, and spectral power, and apply both classical machine learning classifiers (such as SVM, XGBoost, RandomForest, and LightGBM) and a lightweight deep learning model (EEGNet).

Key analyses in this project comprise within-subject binary classification of musical preference, an improved pipeline using feature selection, data augmentation, and hyperparameter tuning, as well as time-resolved decoding to determine when preference becomes decodable. We also investigate interpretability of model predictions using SHAP values with topographic brain maps, the potential for cross-subject generalization, and the impact of familiarity on decoding performance.

## Setup

This project requires Python 3.12+ and the [uv](https://docs.astral.sh/uv/) package manager.

1. To install:

    ```bash
    git clone https://github.com/jktrns/cogs-189.git
    cd cogs-189
    uv sync
    ```

2. The [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) requires registration for academic use:

    1. Visit the [DEAP download page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)
    2. Request access via the [dataset request form](https://anaxagoras.eecs.qmul.ac.uk/request.php?dataset=DEAP)
    3. Download the *preprocessed Python format* (`data_preprocessed_python.zip`, ~2.7GB)
    4. Extract the `.dat` files into `data/deap/`. The `data/deap/` directory should contain `s01.dat` through `s32.dat`.

## References

- Koelstra, S. et al. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. *IEEE Trans. Affective Computing*, 3(1), 18–31.
- Hadjidimitriou, S. & Hadjileontiadis, L. (2012). Toward an EEG-Based Recognition of Music Liking. *IEEE Trans. Biomedical Engineering*, 59(12), 3498–3510.
- He, Z. et al. (2024). Introducing EEG Analyses to Help Personal Music Preference Prediction. *arXiv:2404.15753*.
