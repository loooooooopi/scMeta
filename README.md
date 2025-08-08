# PreMet: Predicting Metastasis sites from the transcriptomes of primary tumors

***

![PreMet Model Overview](./PreMet_model.png)

PreMet is a variational autoencoder (VAE)-based deep learning model specifically designed for metastasis site prediction from transcriptomic profiles of primary tumors. The model employs a supervised latent representation learning framework, where the VAE's encoder maps input gene expression data into a lower-dimensional latent space, and two task-specific deep neural networks predict the primary tumor type and metastasis site. PreMet is trained using a one-vs-rest approach to handle significant class imbalance across metastasis site labels, with focal loss implemented for metastasis prediction to emphasize harder-to-classify samples. 


*** 
## Requirements 
Required packages:
- [Scanpy](https://scanpy.readthedocs.io/en/stable/)
- [anndata](https://anndata.readthedocs.io/en/latest/)
- [Pytorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/index.html)


To directly create a Conda enviroment for PreMet, please run

```
conda env create -f environment.yml
```
***

## Reproducibility:

- [Data pre-processing](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/data_preprocessing.ipynb)
- [Baseline model1: PreMet-DNN](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/Baseline1%20PreMet-DNN.ipynb)
- [Baseline model2: PreMet-Multi](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/Baseline2%20PreMet-Multi.ipynb)
- [PreMet-VAE (main model)](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/PreMet-VAE.ipynb)
- [Gradient-based feature prioritization](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/Gradient-based%20feature%20selection%20and%20pathway%20analysis.ipynb)

The trained best models were provided in [Reproducibility](https://github.com/loooooooopi/PreMet/blob/main/Reproducibility/) in tar.gz for each model.

***

## Train on new data and predict for new samples

Tutorials are provided in: [train new model](https://github.com/loooooooopi/PreMet/blob/main/train_new_data.ipynb)