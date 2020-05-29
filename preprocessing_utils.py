import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split

nominal_columns = ['fn', 'tid', 'title', 'wordsInTitle', 'url']

num_columns = [
    'ratingCount',
    'duration',
    'year',
    'nrOfWins',
    'nrOfNominations',
    'nrOfPhotos',
    'nrOfNewsArticles',
    'nrOfUserReviews',
    'nrOfGenre',
]

comb_columns = ['totalNominations', 'winsPerNomination', 'reviewsPerRating']


def stratify_split_imdb(df, test_size=0.2, drop_cat=True, random_state=42):
    """
    Splits the data into a training set and a test set with stratifying to keep the distributions similar

    If drop_cat is False, the categorical feature used for stratification will be kept in the output datasets
    """

    cat = np.array(pd.cut(df['imdbRating'], bins=10, labels=range(10)))
    if not drop_cat:
        df = df.copy()
        df['imdbRating_cat'] = cat

    training_set, test_set = train_test_split(df, test_size=0.2, stratify=cat, random_state=random_state)
    return training_set, test_set

def separate_features_targets(df, target_column='imdbRating'):
    feature_columns = list(df.columns)
    feature_columns.remove(target_column)
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


class NumericalFeatureCombinator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['totalNominations'] = X['nrOfNominations'] + X['nrOfWins']
        X['winsPerNomination'] = X['nrOfWins'] / X['totalNominations']
        X['winsPerNomination'] = X['winsPerNomination'].fillna(0)
        X['reviewsPerRating'] = X['nrOfUserReviews'] / X['ratingCount']
        X['reviewsPerRating'] = X['reviewsPerRating'].fillna(0)
        return X


class DurationImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_movies = X[X['type'] == 'video.movie']
        self.movie_duration_median = X_movies['duration'].median()

        X_tv_shows = X[X['type'] == 'video.tv']
        self.tv_show_duration_median = X_tv_shows['duration'].median()
        
        X_episodes = X[X['type'] == 'video.episode']
        self.episode_duration_median = X_episodes['duration'].median()
        return self
    
    def transform(self, X):
        X.loc[X['type'] == 'game', 'duration'] = 0

        movies_condition = X['type'] == 'video.movie'
        X.loc[movies_condition, 'duration'] = X.loc[movies_condition, 'duration'].fillna(self.movie_duration_median)

        tv_shows_condition = X['type'] == 'video.tv'
        X.loc[tv_shows_condition, 'duration'] = X.loc[tv_shows_condition, 'duration'].fillna(self.tv_show_duration_median)

        episodes_condition = X['type'] == 'video.episode'
        X.loc[episodes_condition, 'duration'] = X.loc[episodes_condition, 'duration'].fillna(self.episode_duration_median)

        return X


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Column transformer for pandas DataFrame objects used to keep column names
    """
    def __init__(self, transformers, remainder='drop'):
        self.transformers_ = transformers
        self._remainder = remainder
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # fit each transformer to its columns
        fit_columns = []
        for name, transformer, columns in self.transformers_:
            if transformer == 'drop' or transformer =='passthrough':
                fit_columns += columns
                continue
            else:
                fit_columns += columns
                transformer.fit(X[columns], y)

        # add remaining columns
        if self._remainder != 'drop':
            remaining_columns = []
            for col in X.columns:
                if col not in fit_columns:
                    remaining_columns.append(col)
            if len(remaining_columns) > 0:
                self.transformers_.append(('remainder', self._remainder, remaining_columns))
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # apply each transformer to its columns
        result_X = None
        for name, transformer, columns in self.transformers_:
            if transformer == 'drop':
                continue
            elif transformer == 'passthrough':
                new_X = X[columns].copy()
            else:
                tr_X_cols = transformer.transform(X[columns])
                if not isinstance(tr_X_cols, pd.DataFrame):
                    if hasattr(tr_X_cols, "todense"):
                        # convert sparse array to dense array
                        tr_X_cols = tr_X_cols.todense()
                    if hasattr(transformer, 'get_feature_names'):
                        # deduce column names from transformer output feature names
                        column_names = transformer.get_feature_names(input_features=columns)
                        tr_X_cols = pd.DataFrame(tr_X_cols, columns=column_names, index=X.index)
                    elif tr_X_cols.shape[1] == len(columns):
                        tr_X_cols = pd.DataFrame(tr_X_cols, columns=columns, index=X[columns].index)
                    else:
                        raise TypeError("Failed to convert numpy array to pandas DataFrame, "
                                        "no get_feature_names attribute in estimator and "
                                        "mismatch in number of columns")
                new_X = tr_X_cols

            # concatenate to previous output columns
            if result_X is None:
                result_X = new_X
            else:
                result_X = pd.concat([result_X, new_X], axis=1)
        return result_X
    
    def get_params(self, deep=True):
        return {"transformers": self.transformers_}


class FeaturePreprocessor(Pipeline):
    """DataFrameCDataFrameColumnTransformerolumnTransformer
    Customizable feature preprocessing pipeline
    """
    def __init__(self, add_combinations=False, powertransform_num=False, std_scale_num=False, onehot_type=False, drop_features=None):
        self.add_combinations = add_combinations
        self.powertransform_num = powertransform_num
        self.std_scale_num = std_scale_num
        self.onehot_type = onehot_type
        self.drop_features = drop_features
        
        steps = []
        num_cols = num_columns.copy()

        # impute the duration
        steps.append(('impute_duration', DurationImputer()))

        # add numerical feature combinations
        if add_combinations:
            steps.append(('combine_num', NumericalFeatureCombinator()))
            num_cols += comb_columns

        col_transformers = [
                            ('drop_nominal', 'drop', nominal_columns),
        ]

        # add transformations to numerical columns
        num_transformer = 'passthrough'
        num_tr_list = []

        # power transform (makes data more Gaussian-like)
        if powertransform_num:
            num_tr_list.append(('powertransform', PowerTransformer()))

        # standard scaling
        if std_scale_num:
            num_tr_list.append(('std_scale', StandardScaler()))
        
        if len(num_tr_list) > 0:
            num_transformer = Pipeline(num_tr_list)
        col_transformers.append(('num_tr', num_transformer, num_cols))
        
        # apply one-hot encoding to the "type" feature
        if onehot_type:
            col_transformers.append(('onehot_type', OneHotEncoder(), ['type']))
        else:
            col_transformers.append(('pass_type', 'passthrough', ['type']))

        steps.append(('col_tr', DataFrameColumnTransformer(col_transformers, remainder='passthrough')))
        
        # drop specified after all other steps have been processed
        if drop_features is not None:
            steps.append(('drop', DataFrameColumnTransformer([('drop', 'drop', drop_features)], remainder='passthrough')))
        
        super().__init__(steps)
    
    def get_params(self, deep=True):
        params_dict = {
            "add_combinations": self.add_combinations,
            "powertransform_num": self.powertransform_num,
            "std_scale_num": self.std_scale_num,
            "onehot_type": self.onehot_type,
            "drop_features": self.drop_features
            }
        params_dict.update(super().get_params(deep=deep))
        return params_dict