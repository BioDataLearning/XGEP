# XGEP for Expression-based Prediction of Human Essential Genes and Associated LncRNAs in Cancer Cells
XGEP is developed to predict essential genes and associated lncRNAs in cancer cells using relevant features derived from the TCGA transcriptome dataset through collaborative embedding. XGEP has been evaluated on pan-cancer data and cancer-type-specific data and used for predicting core essential genes and essential genes in specific cancer types. It is implemented in Python.

This documentation is part of the supplementary information release for XGEP. For details of this work, please refer to our paper "Expression-based Prediction of Human Essential Genes and Associated LncRNAs in Cancer Cells" (S. Kuang, Y. Wei and L. Wang, 2020).

## Requirements
- python3
- argparse
- numpy 
- pandas
- sklearn
- xgboost
- keras >=2.0
- tensorflow
- csv
- hyperopt
- fastai

## Embedding Vectors
The gene embedding vectors for essential and non-essential genes have been deposited into the Data directory. These embedding vectors were generated from the transcriptome data of either pan-cancer or specific cancer types (COAD, CESC and GBM) using unsupervised collaborative embedding. The transcriptome data were downloaded from TCGA database (https://portal.gdc.cancer.gov/). The code for collaborative embedding were modified from https://github.com/zeochoy/tcga-embedding/blob/master/train.py. If you would like to generate your own embedding vectors using the code, you will need to install fastai library.

## Hyperparameter Optimization of DNN models
Gradient boosted tree (implemented using XGBoost, called as XGB in this study), support vector machine (SVM) and deep neural network (DNN) models were included in XGEP to predict gene essentiality in cancer cells. Considering the relatively large hyperparameter space of DNN models, Bayesian optimization, which has been shown to be more time-efficient for hyperparameter tuning than grid or random search, was used to hyperparameter tuning of the DNN models. We have included the optimized hyperparameters used in our study in "XGEP.py". If you would like to run the hyperparameter tuning by yourself, please use the following command line:

```
$ python DNN_hyperparameters_search_test.py -e Essential_genes_pancancer_emb_len150 -n Non_essential_genes_pancancer_emb_len150 -o DNN_hyperparamter_output
```

Please substitute the 'Essential_genes_pancancer_emb_len150_test' and 'Non_essential_genes_pancancer_emb_len150' with your own embedding vectors for essential (positive) and non-essential (negative) genes.

## XGEP Evaluation and Essential Gene Prediction
To evaluate the performance of XGEP and make prediction on the genes not included in the training dataset, the following command line could be used:

```
$ python XGEP.py -e Essential_genes_pancancer_emb_len150 -n Non_essential_genes_pancancer_emb_len150 -u Genes_for_prediction_pancancer_emb_len150 -o performance_output -p predicted_probability.csv
```

Here, 'Essential_genes_pancancer_emb_len150', 'Non_essential_genes_pancancer_emb_len150' and 'Genes_for_prediction_pancancer_emb_len150' are the gene embedding vectors for essential genes (positive instances), non-essential genes (negative instances) and genes not included in the training data. Please substitute them with your own data.

The predicted_probability.csv file will store the predictions and predicted probability of all the three models. For example,

|Gene|SVM_prediction|SVM_pred_prob|XGB_prediction|XGB_pred_prob|DNN_prediction|DNN_pred_prob|
|----|:-------------|:------------|:-------------|:------------|:-------------|:------------|
|ENSG00000242268.2|0|0.0036|0|3.5686e-05|0|0.0057|
|ENSG00000179632.8|1|0.5866|1|0.6802|1|0.8802|
|ENSG00000134717.16|1|0.6268|0|0.2164|1|0.8524|


