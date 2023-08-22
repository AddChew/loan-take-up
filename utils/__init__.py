import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
import plotly.io as pio
import plotly.express as px

from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


pio.renderers.default = 'plotly_mimetype+notebook'


lgb_parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'seed': 0,
}


def standardize_column_names(df: pd.DataFrame, 
                   custom_mapping: dict = {'CCAvgSpending': 'cc_avg_spending', 'InternetBanking': 'internet_banking'}
                   ) -> pd.DataFrame:
    """Standardize columns naming convention to lowercase, with spaces denoted by _.

    Args:
        df (pd.DataFrame): dataframe with columns to rename.
        custom_mapping (dict, optional): dictionary containing columns that require custom mapping as keys and their 
            corresponding mapped values as values. Defaults to {'CCAvgSpending': 'cc_avg_spending', 'InternetBanking': 'internet_banking'}.

    Returns:
        pd.DataFrame: dataframe with renamed columns.
    """
    return df.rename(
        columns = lambda x: x.lower().replace(' ', '_') if x not in custom_mapping else custom_mapping[x]
    )


def check_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Check if any of the columns in the dataframe contains missing values and calculate the missing percentage.

    Args:
        df (pd.DataFrame): dataframe to check.

    Returns:
        pd.DataFrame: Dataframe consisting of two columns:
            - first column consists of boolean values to denote whether the column contains missing values.
            - second column contains the missing count of the column.
            - third column contains the missing percentage of the column.
    """
    is_null = df.isnull()
    return pd.DataFrame({
        'contains_missing': is_null.any(),
        'missing_count': is_null.sum(),
        'missing_percent': 100 * is_null.sum() / df.shape[0]
    })


def plot_categories_distribution(df: pd.DataFrame, category_col: str, title: str = None, width: int = 500, height: int = 800):
    """Plot distribution of categorical values.

    Args:
        df (pd.DataFrame): dataframe containing categorical columns.
        category_col (str): name of categorical column.
        title (str): title of plot.
        width (int, optional): width of plot. Defaults to 500.
        height (int, optional): height of plot. Defaults to 800.

    Returns:
        Figure: distribution plot of categorical values.
    """
    title = f'Percentage of each {category_col} in dataset' if title is None else title
    fig = px.bar(
        (df[category_col].astype(str).value_counts() / df.shape[0] * 100).sort_values().reset_index().rename(columns = {'count': 'percentage'}), 
        x = 'percentage', y = category_col, title = title
    )
    fig.update_yaxes(tickmode = 'linear')
    fig.update_layout(width = width, height = height)
    return fig


def compute_loan_probability(df: pd.DataFrame, category_col: str, 
                             dummy_col: str = 'id', label_col: str = 'personal_loan',
                             ascending: bool = True,
                             ) -> pd.DataFrame:
    """Compute loan probability given categorical value.

    Args:
        df (pd.DataFrame): dataframe containing category column.
        category_col (str): name of category column.
        dummy_col (str, optional): dummy column to use for groupby. Defaults to 'id'.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.
        ascending (bool, optional): flag to indicate whether to sort probabilities in ascending order.
            Defaults to True.

    Returns:
        pd.DataFrame: dataframe containing the computed probabilities.
    """
    group_cols = [category_col, label_col]
    required_cols = group_cols + [dummy_col]
    renamed_col = 'probability'

    df_probs = df[required_cols].groupby(group_cols).count() / df[[category_col, dummy_col]].groupby([category_col]).count()
    df_probs = df_probs.reset_index().rename(columns = {dummy_col: renamed_col})
    df_probs = df_probs[df_probs[label_col] == 1].sort_values(by = renamed_col, ascending = ascending)

    return df_probs


def plot_category_loan_distribution(df: pd.DataFrame, category_col: str, dummy_col: str = 'id', 
                                    label_col: str = 'personal_loan', height: int = 800, 
                                    width: int = 600, title: str = None):
    """Plot probability of loan take up given categorical value.

    Args:
        df (pd.DataFrame): dataframe containing category column.
        category_col (str): name of category column.
        dummy_col (str, optional): dummy column to use for groupby. Defaults to 'id'.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.
        height (int, optional): height of figure. Defaults to 800.
        width (int, optional): width of figure. Defaults to 600.
        title (str, optional): figure title. Defaults to None.

    Returns:
        Figure: plot of probability given categorical value.
    """
    title = f'Probability of {label_col} = 1 given {category_col}' if title is None else title
    df_probs = compute_loan_probability(df, category_col, dummy_col, label_col)
    df_probs[category_col] = df_probs[category_col].astype(str)

    fig = px.bar(df_probs, y = category_col, x = 'probability', height = height, width = width, title = title)
    fig.update_yaxes(tickmode = 'linear')
    return fig


def encode_categorical_features(df: pd.DataFrame,
                                categorical_features: list = [
                                    'postal_code', 'education', 'investment_account', 'deposit_account', 'internet_banking'
                                ], 
                                encoder_path: str = './encoder/encoders.pkl') -> tuple:
    """Encode categorical feature values to numerical values.

    Args:
        df (pd.DataFrame): dataset containing categorical features to encode.
        categorical_features (list, optional): names of categorical feature columns. 
            Defaults to ['postal_code', 'education', 'investment_account', 'deposit_account', 'internet_banking'].
        encoder_path (str, optional): file path to save encoder dict. Defaults to './encoder/encoders.pkl'.

    Returns:
        tuple: encoded dataset, encoder_dict
    """
    encoder_dict = {}

    for feature in categorical_features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
        encoder_dict[feature] = encoder

    joblib.dump(encoder_dict, encoder_path)
    return df, encoder_dict


def train_lgb_model(train: pd.DataFrame, num_boost_rounds: int, features: list, 
                    cat_features: list, val: pd.DataFrame = None, label_col: str = 'personal_loan',
                    lgb_parameters: dict = lgb_parameters, model_path: str = None,
                    callbacks: list = None) -> lgb.Booster:
    """Train lightGBM model.

    Args:
        train (pd.DataFrame): dataframe containing train dataset.
        num_boost_rounds (int): number of boosting rounds to train model.
        features (list): full list of features used for model training.
        cat_features (list): list of categorical features used for model training.
        val (pd.DataFrame, optional): dataframe containing validation dataset. Defaults to None.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.
        lgb_parameters (dict, optional): training hyperparameters. Defaults to lgb_parameters.
        model_path (str, optional): file path to save model to. Defaults to None.
        callbacks (list, optional): list of callbacks to apply during model training. Defaults to None.

    Returns:
        lgb.Booster: fitted model.
    """
    train_data = lgb.Dataset(train[features], categorical_feature = cat_features, label = train[label_col], free_raw_data = False)
    val_data = None

    if val is not None:
        val_data = lgb.Dataset(val[features], categorical_feature = cat_features, label = val[label_col], free_raw_data = False)

    model = lgb.train(
        lgb_parameters, train_data, valid_sets = val_data, num_boost_round = num_boost_rounds,
        callbacks = callbacks
    )
    if model_path is not None:
        model.save_model(model_path)

    return model


def train_catboost_model(train: pd.DataFrame, num_boost_rounds: int, features: list, 
                    cat_features: list, val: pd.DataFrame = None, label_col: str = 'personal_loan',
                    stopping_rounds: int = None, model_path: str = None) -> CatBoostClassifier:
    """Train catboost model.

    Args:
        train (pd.DataFrame): dataframe containing train dataset.
        num_boost_rounds (int): number of boosting rounds to train model.
        features (list): full list of features used for model training.
        cat_features (list): list of categorical features used for model training.
        val (pd.DataFrame, optional): dataframe containing validation dataset. Defaults to None.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.
        stopping_rounds (int, optional): number of early stopping rounds. Defaults to None.
        model_path (str, optional): file path to save model to. Defaults to None.

    Returns:
        CatBoostClassifier: fitted catboost model.
    """
    val_data = None
    model = CatBoostClassifier(random_state = 0, num_boost_round = num_boost_rounds)

    if val is not None:
        val_data = (val[features], val[label_col])

    model.fit(
        X = train[features], y = train[label_col],
        eval_set = val_data, cat_features = cat_features,
        early_stopping_rounds = stopping_rounds,
    )
    if model_path is not None:
        model.save_model(model_path)

    return model


def nested_stratified_kfold_cv(train: pd.DataFrame, features: list, cat_features: list, 
                               model_name: str, num_folds: int = 5, threshold: float = 0.5, 
                               label_col: str = 'personal_loan', scaling_factor: float = 1,
                               stopping_rounds: int = 5) -> dict:
    """Perform nested stratified k-fold cross-validation.

    Args:
        train (pd.DataFrame): train dataframe.
        features (list): full list of features used for model training.
        cat_features (list): list of categorical features used for model training.
        model_name (str): name of model to use. Takes on the value 'catboost' or lightgbm'.
        num_folds (int, optional): number of folds for cross validation. Defaults to 5.
        threshold (float, optional): probability threshold for predicting personal_loan = 1 class. Defaults to 0.5.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.
        scaling_factor (float, optional): multiplier to use for number of boosting rounds. Defaults to 1.
        stopping_rounds (int, optional): number of early stopping rounds. Defaults to 5.

    Returns:
        dict: stratified kfold results.
    """
    skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 0)

    # Initialize metrics/hyperparameters containers
    val_f1_list = []
    test_f1_list = []
    val_num_boost_rounds_list = []
    test_num_boost_rounds_list = []

    # Outer loop splits the dataset into train (N - 1 partitions) and test (1 partition) datasets
    for train_idx_, test_idx in skf.split(train, train[label_col]):
        train_outer = train.iloc[train_idx_]
        test_outer = train.iloc[test_idx]

        print(f'Outer train shape: {train_outer.shape}')
        print(f'test shape: {test_outer.shape}')

        inner_num_boost_rounds_list = []

        # Inner loop splits the train dataset (N - 1 partitions) further into a smaller train (K - 1 partitions) and validation (1 partition) dataset for hyperparameter tuning
        for train_idx, val_idx in skf.split(train_outer, train_outer[label_col]):
            train_inner = train_outer.iloc[train_idx]
            val_inner = train_outer.iloc[val_idx]

            print(f'Inner train shape: {train_inner.shape}')
            print(f'val shape: {val_inner.shape}')

            if model_name == 'lightgbm':
                model = train_lgb_model(
                    train = train_inner, num_boost_rounds = 100, 
                    features = features, cat_features = cat_features,
                    val = val_inner, callbacks = [lgb.early_stopping(stopping_rounds = stopping_rounds)]
                )
                best_iter = model.best_iteration

            else:
                model = train_catboost_model(
                    train = train_inner, num_boost_rounds = 100,
                    features = features, cat_features = cat_features,
                    val = val_inner, stopping_rounds = stopping_rounds
                )
                best_iter = model.best_iteration_

            print(f'Best iteration: {best_iter}')
            val_preds = model.predict(val_inner[features]) > threshold
            val_f1 = f1_score(val_inner[label_col], val_preds)
            print(f'Validation f1 score: {val_f1}')
            
            val_f1_list.append(val_f1)
            inner_num_boost_rounds_list.append(best_iter)
            val_num_boost_rounds_list.append(best_iter)

        # Train on full N - 1 partitions dataset and test on 1 partition held-out test set
        # Extrapolate the num_boost_rounds based on the ratio K / (K - 1)
        avg_val_num_boost_rounds = np.mean(inner_num_boost_rounds_list)
        print(f'Average best iteration: {avg_val_num_boost_rounds}')

        scaled_num_boost_rounds = int(avg_val_num_boost_rounds * scaling_factor)
        print(f'Scaled average best iteration: {scaled_num_boost_rounds}')

        if model_name == 'lightgbm':
            model = train_lgb_model(
                train = train_outer, num_boost_rounds = scaled_num_boost_rounds,
                features = features, cat_features = cat_features
            )
        else:
            model = train_catboost_model(
                train = train_outer, num_boost_rounds = scaled_num_boost_rounds,
                features = features, cat_features = cat_features
            )

        test_preds = model.predict(test_outer[features]) > threshold
        test_f1 = f1_score(test_outer[label_col], test_preds)
        print(f'Test f1 score: {test_f1}')
        test_f1_list.append(test_f1)
        test_num_boost_rounds_list.append(scaled_num_boost_rounds)

    return {
        'val_f1': val_f1_list,
        'test_f1': test_f1_list,
        'val_num_boost_rounds': val_num_boost_rounds_list,
        'test_num_boost_rounds': test_num_boost_rounds_list
    }