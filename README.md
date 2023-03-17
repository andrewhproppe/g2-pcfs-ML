This repository can be used for generating synthetic data by simulating photon correlation Fourier spectroscopy (PCFS) experiments, and using the datasets to traing Adversarial Autoencoder Ensemble (AAE) models.

To generate training data, use the script make_devset.py in /data/raw/, specificying the number of PCFS experiments to simulate and the degree of data augmentation.

For model training, we used the Weights & Biases package to log loss and plot predicted g2s. We used Google Colab to perform model training on GPUs, and provide the annotated notebook we used for training here: https://colab.research.google.com/drive/1b5IQ3obw-pGtpNfad-8h3Xgdg4EvGjwW,
