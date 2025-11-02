Project Structure and Overview
===========================
This document provides an overview of the project structure and its components.
Project Structure

-----------------
The project is organized into the following main directories and files:
- `data/`: Contains datasets used for training and evaluation.
- `output/`: Stores generated outputs, models, and logs.
- `batch_files/`: Contains batch scripts for automating tasks.

The following files are included in the project:
- `auto_enc.py`: Main script for the autoencoder model. It includes  various forms of autoencoders such as Linear Autoencoder, Non-Linear Autoencoder, and Autoencoder with masked loss function.
- `tp_apc.py`: Script for training and evaluating the TP-APC model (Cahat et al., 2023).
- `tw_apc.py`: Script for training and evaluating the TW-APC model.
- `snp_data_downloader.py`: Script for downloading SNP data from public databases.

The following files are used for running simulations and they also serve as example scripts to run models:
- `synthetic_data_analysis.py`: Running simulations for imputation on synthetically generated data with a strong factor structure. Missingness pattern is fully random.
- `synthetic_data_analysis_blocks.py`: Running simulations for imputation on synthetically generated data. Missingness pattern is non-random.
- `snp_analysis.py`: Running simulations for SNP data imputation.
- `bank_analysis_taiwan.py`: Running simulations for Taiwan bank data imputation. Missing points are artificially generated
- `bank_data_analysis_kaggle.py`: Running simulations for imputation on banking data from Kaggle. 