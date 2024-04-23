This repository is accompanying a conference report.

The file named LLM represent the code that was applied to train and test a Large Language Model. The file named Random Forest contains the code that trained and tested a Random Forest Classifier.

There are three datasets: df_preprocessed is used to train and test an LLM model; the other two datasets (train and test) are used to train and test a Random Forest model. These datasets (train and test) are derived from a train-test split, conducted in the LLM code. To preserve objective metric estimations the LLM train-test datasets are saved to be used for the Random Forest algorithm.
