# scMeta: Predicting Metastasis Potential at Single-cell Resolution

***

![scMeta Overview](./scMeta.png)

scMeta is a graph transformer–based deep learning framework for predicting metastatic potential at single-cell resolution from transcriptomic data. It constructs a cell–cell graph based on transcriptional similarity, learns context-aware embeddings via attention-based message passing, and uses these embeddings for metastasis classification, biomarker prioritization, and pathway enrichment analysis.

*** 
## Requirements 
Required packages:
- [Scanpy](https://scanpy.readthedocs.io/en/stable/)
- [Anndata](https://anndata.readthedocs.io/en/latest/)
- [Pytorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/index.html)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [Harmony-Pytorch](https://github.com/lilab-bcb/harmony-pytorch) (for integration only)

## Reproducibility:

- [Data pre-processing](https://github.com/loooooooopi/scMeta/tree/master/Pre-processing)
  - The four Pre-processing notebooks contains all information for inidividual studies, including the source for raw data and annotations.
  - Final integration of all datasets was in Integrate all data.ipynb.
  - Note: The raw and processed data used in training scMeta and trained best models will be deposited to Zenodo soon.
- [Baseline model](https://github.com/loooooooopi/scMeta/blob/master/Reproducibility/baseline_models.ipynb)
  - This notebook contains the code for 5 fold CV and LOOCV using 3 classical machine learning models as baselines.
- [scMeta (main model)](https://github.com/loooooooopi/scMeta/tree/master/Reproducibility)
  - [5 fold CV notbook](https://github.com/loooooooopi/scMeta/blob/master/Reproducibility/scMeta_5foldCV.ipynb)
  - [LOOCV notbook](https://github.com/loooooooopi/scMeta/blob/master/Reproducibility/scMeta_LOOCV.ipynb)
- [Downstream analysis including feature priorization and pathway analysis](https://github.com/loooooooopi/scMeta/blob/master/Reproducibility/scMeta_downstream_analysis.ipynb)


***

## Train on new data and predict for new samples

Tutorials are provided in: [train new model](https://github.com/loooooooopi/scMeta/blob/master/train_scMeta.ipynb)