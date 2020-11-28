## Predicting IMDb Ratings
I used a publicly available dataset from Kaggle to train and evaluate user rating prediction models using Scikit Learn and Keras.

I followed the standard Machine Learning process and separated the steps in Jupyter notebook files:
- Get the data (get_data.ipynb)
- Exploring and visualizing the data (data_analysis.ipynb)
- Cleaning and processing the data for machine learning models (data_preparation.ipynb)
- Evaluating different machine learning models on the data (shortlist_models.ipynb)
- Finetuning a few models and choosing the best (finetuning_forest.ipynb, finetuning_neuranet.ipynb)

The best model so far is the finetuned fully-connected neural network achieveing an MSE of about 0.781.