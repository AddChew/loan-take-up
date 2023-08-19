import joblib
import pandas as pd
import plotly.io as pio

import plotly.express as px
from sklearn.preprocessing import LabelEncoder


pio.renderers.default = 'plotly_mimetype+notebook'


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
                             dummy_col: str = 'id', label_col: str = 'personal_loan'
                             ) -> pd.DataFrame:
    """Compute loan probability given categorical value.

    Args:
        df (pd.DataFrame): dataframe containing category column.
        category_col (str): name of category column.
        dummy_col (str, optional): dummy column to use for groupby. Defaults to 'id'.
        label_col (str, optional): name of label column. Defaults to 'personal_loan'.

    Returns:
        pd.DataFrame: dataframe containing the computed probabilities.
    """
    group_cols = [category_col, label_col]
    required_cols = group_cols + [dummy_col]
    renamed_col = 'probability'

    df_probs = df[required_cols].groupby(group_cols).count() / df[[category_col, dummy_col]].groupby([category_col]).count()
    df_probs = df_probs.reset_index().rename(columns = {dummy_col: renamed_col})
    df_probs = df_probs[df_probs[label_col] == 1].sort_values(by = renamed_col)

    return df_probs


def plot_category_loan_distribution(df: pd.DataFrame, category_col: str, dummy_col: str = 'id', 
                                    label_col: str = 'personal_loan', height: int = 800, 
                                    width: int = 600, title: str = None):
    """Plot probability of loan conversion given categorical value.

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