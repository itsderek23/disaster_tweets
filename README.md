# Real or Not? NLP with Disaster Tweets

This is a Tensorflow-backed Keras model that predicts which tweets are about real disasters and which ones are not. It's derived from the popular [Basic EDA,Cleaning and GloVe Kaggle Notebook](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove).

The project is structured with [whisk](https://github.com/whisk-ml/whisk), an ML project framework that make makes collaboration, reproducibility, and deployment "just work".

Besides Tensorflow+Keras, the project uses DVC to version control the data download and training stages. As the training stage takes ~20 minutes on a laptop, this can save a significant amount of time when bootstrapping the project.

## Prerequisites

The following is required to run this project:

* Git
* Python 3.6+
* A Linux-based OS

## Setup

After cloning this repo and `cd disaster_tweets` run the following in your terminal:

    pip install whisk
    whisk setup
    source venv/bin/activate
    whisk dvc setup
    dvc pull

The commands above install whisk, setup the project environment, activate the created venv, setup dvc, and download data stored in DVC.

## Quickstart

After running setup, invoke the model from the command line:

```
disaster_tweets predict '["Theyd probably still show more life than Arsenal did yesterday, eh? EH?"]'
[[0.19104013]]

disaster_tweets predict '["Just happened a terrible car crash"]'
[[0.658098]]
```

### DVC stages

Run the training stage:

    dvc repro train.dvc

Run the download stages:

    dvc repro download_dataset.dvc
    dvc repro download_glove.dvc

## Learn more about whisk

To learn more about whisk, here are a few helpful doc pages:

* [A Quick Tour of whisk](https://whisk.readthedocs.io/en/latest/tour_of_whisk.html)
* [Key Concepts](https://whisk.readthedocs.io/en/latest/key_concepts.html)
* [Project Structure](https://whisk.readthedocs.io/en/latest/project_structure.html)
