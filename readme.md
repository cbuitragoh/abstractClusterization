# Abstracts Clusterization

This project applied simple clustering with K-Means to find common themes in the NSF research award abstracts dataset.


## Installation

Follow these steps to install the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/cbuitragoh/abstractClusterization.git
   cd abstractClusterization

## Use 

2. Download NSF Research Awards Abstracts dataset on your working machine
3. create virtual env and install all dependencies from requirements.txt
4. create .env file in notebooks, pipelines and tests folders to set environment variables needs in project
5. Create data and models folder in root directory
6. set these environment variables in all .env files: 
    - DATA_PATH:the path where you downloaded the dataset
    - CSV_DATA_PATH: The full path to the file to save the abstract_narration.csv file when running the eda notebook (this file inside data folder)
    - CLUSTERED_ASBTRACTS_PATH: The full path to the file to save the clustered_abstracts.csv file when running the pipelines/training.py (this file inside data folder)
    - MODELS_PATH=The full path to the models folder to save .pkl files when running the pipelines/training.py
    
5. Please review the eda in the notebooks directory to understand the basics of the dataset and how to use .env file
6. Run pipeline/training.py file to train kmeans model (The first time, the all-MiniLM-L6-v2 model is downloaded)
7. Run pipeline/inference.py model to make predictions

## Optional

8. Run src/app.py to run fastAPI app to make prediction on local environment
9. Create a docker image using dockerfile to deploy fastAPI app with containers

## Notes
This repo run GitHubActions from test_trigger.yaml file
Tests are performed on individual functions from the training.py, inference.py files. Python style (PEP8) is also evaluated using flake8