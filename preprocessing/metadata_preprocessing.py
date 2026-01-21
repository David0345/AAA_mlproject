import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import preprocessing.text_preprocessing as text_preprocessing


class MultiTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder, позволяющий закодировать сразу  >=2 столбцов по >=2 таргетам

    Parameters
    --------
        cat_cols (list): list of columns that should be encoded
        target_cols (list): list of target columns
        encoder_cls (TargetEncoder/...): encoder class
        encoder_params (dict): parameters for encoder

    Methods
    --------
        fit(X: pd.DataFrame, y: pd.DataFrame) -> self:
            training encoders
        transform(X: pd.DataFrame) -> pd.DataFrame:
            transform data(df)
    """
    def __init__(self, cat_cols, target_cols, encoder_cls, encoder_params=None):
        self.cat_cols = cat_cols
        self.target_cols = target_cols
        self.encoder_cls = encoder_cls
        self.encoder_params = encoder_params or {}
        self.encoders_ = None
        self.X_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.encoders_ = {}
        self.X_ = X

        for target in self.target_cols:
            enc = self.encoder_cls(**self.encoder_params)
            enc.fit(X[self.cat_cols], y[target])
            self.encoders_[target] = enc

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded_parts = []

        # iterating to encode by every encoder
        for target, enc in self.encoders_.items():
            X_enc = enc.transform(X[self.cat_cols])

            X_enc = pd.DataFrame(
                X_enc,
                columns=[
                    f'{col}_te_{target}'
                    for col in self.cat_cols
                ],
                index=X.index
            )

            encoded_parts.append(X_enc)

        X = X.drop(self.cat_cols, axis=1)
        encoded_parts = pd.concat(encoded_parts, axis=1)
        return pd.concat([encoded_parts, X], axis=1)


def drop_outliners(df: pd.DataFrame) -> pd.DataFrame:
    """
    getting rid of outliners (top 100 highest by every target)

    params
        df (pd.DataFrame): dataframe

    returns:
        df withoit outliners
    """
    heavy_weights = df.nlargest(100, 'real_weight')
    highest = df.nlargest(100, 'real_height')
    longest = df.nlargest(100, 'real_length')
    widthest = df.nlargest(100, 'real_width')

    indexes_to_drop = pd.concat(
        [
            pd.Series(heavy_weights.index),
            pd.Series(highest.index),
            pd.Series(longest.index),
            pd.Series(widthest.index)
        ]
    )
    indexes_to_drop.to_csv('data/indexes_to_ignore.csv', index=False)
    df = df.drop(indexes_to_drop)
    return df


def nan_to_cat(df: pd.DataFrame, cat) -> pd.DataFrame:
    """
    impute all Nans in category column with new column - 'no info'

    params
        df (pd.DataFrame): dataframe
        cat (str): category column name

    returns:
        df with imputed cat column
    """
    df.loc[df[cat].isna(), cat] = 'no_info'
    return df


def cyclic_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    encoding timeseries data by months

    params
        df (pd.DataFrame): dataframe
        cat (str): category column name

    returns:
        df where 'order_data' column replaces with two sin-wise and cos-wise columns
    """
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['month_of_purchase'] = df['order_date'].dt.month
    df['month_sin_encoded'] = np.sin(2 * np.pi * df['month_of_purchase']/12.0)
    df['month_cos_encoded'] = np.cos(2 * np.pi * df['month_of_purchase']/12.0)
    df.drop(['order_date', 'month_of_purchase'], axis=1, inplace=True)
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    drops irrelevant columns from df

    params
        df (pd.DataFrame): dataframe

    returns:
        df with dropped columns
    """
    cats_to_drop = ['item_id', 'seller_id', 'buyer_id', 'image_name', 'title', 'description', 'image_name']
    df = df.drop(cats_to_drop, axis=1, inplace=True)
    return df


class MultiStepPreprocessor:
    """
    Multstepprocessor for training regression model.

    Multiprocessor including full cycle of processing data:
    - Imputing nans, encoding categorical features, drop outliners
    - Getting fasttext embeddings for every text
    - Loading and merging image embeddings
    - Automatic splitting train/val data

    fit-transform interface

    Attributes
    ----------
    condition_encoder (ColumnTransformer or None):
        OneHotEncoder for item_condition column
    multiple_target_encoder (MultiTargetEncoder or None):
        TargetEncoder for cat_cols
    text_processor (TextPreprocessor or None):
        FastText + TF-IDF preprocessor
    cat_cols (list):
        category feats for target encoding
        ['category_name', 'subcategory_name', 'microcat_name']
    target_cols (list):
        target columns
        ['real_weight', 'real_height', 'real_length', 'real_width']
    cols_to_drop (list or None):
        columns to drop
        ['image_name', 'title', 'description', 'buyer_id', 'seller_id']

    Methods
    -------
    fit_transform(df_train: pd.DataFrame, train_image_path: str = 'image_embeddings/train_embeddings.parquet') -> tuple
        train all encoders on train data and returns train/val data

        Parameters:
        df_train : pd.DataFrame
        train_image_path : str, optional
        Default: 'image_embeddings/train_embeddings.parquet'

        Returns:
        tuple: (X_train, X_val, y_train, y_val)
        X_train : pd.DataFrame
        X_val : pd.DataFrame
        y_train : pd.DataFrame
        y_val : pd.DataFrame

    transform(df_test: pd.DataFrame, test_image_path: str = 'image_embeddings/test_embeddings.parquet') -> pd.DataFrame
        using all encoders of test data

        Parameters:
        df_test : pd.DataFrame
        test_image_path : str, optional
        Default: 'image_embeddings/test_embeddings.parquet'

        Returns:
        pd.DataFrame

        Raises:
        ValueError
            if transform used before fit_transform

    _basic_preprocessing(df: pd.DataFrame, to_train: bool = True) -> pd.DataFrame
        private method, preprocessing method

    _prepare_text_features(df: pd.DataFrame, to_train: bool) -> np.ndarray
        private method, gettind text features

    Examples
    --------
    >>> import pandas as pd
    >>> from metadata_preprocessing import MultiStepPreprocessor
    >>>
    >>> # Загрузка данных
    >>> df_train = pd.read_parquet('train.parquet')
    >>> df_test = pd.read_parquet('test.parquet')
    >>>
    >>> # Создание и обучение препроцессора
    >>> preprocessor = MultiStepPreprocessor()
    >>> X_train, X_val, y_train, y_val = preprocessor.fit_transform(df_train)
    >>>
    >>> # Обработка тестовых данных
    >>> X_test = preprocessor.transform(df_test)
    >>>
    >>> # Обучение модели
    >>> model.fit(X_train.drop('item_id', axis=1), y_train)
    >>> predictions = model.predict(X_test.drop('item_id', axis=1))

    """
    def __init__(self):
        self.condition_encoder = None
        self.multiple_target_encoder = None
        self.text_processor = None
        self.cat_cols = ['category_name', 'subcategory_name', 'microcat_name']
        self.target_cols = ['real_weight', 'real_height', 'real_length', 'real_width']

    def _basic_preprocessing(self, df, to_train=True):
        """basic preprocessing"""
        if to_train:
            df = drop_outliners(df)
        df = nan_to_cat(df, 'item_condition')
        df = cyclic_encoding(df)
        return df

    def _prepare_text_features(self, df: pd.DataFrame, to_train: bool) -> np.ndarray:
        """preparing text features"""
        texts = df[['title', 'description']].copy()

        if to_train:
            self.text_processor = text_preprocessing.TextPreprocessor()
            texts = self.text_processor.fit_transform(texts)
        else:
            texts = self.text_processor.transform(texts)

        return texts

    def fit_transform(self, df_train, train_image_path='image_embeddings/train_embeddings.parquet'):
        """Обучение и преобразование train данных"""

        # basic preprocessing
        df_train = self._basic_preprocessing(df_train)

        # splitting on feats/targets
        target_df = df_train[self.target_cols]
        feats_df = df_train.drop(self.target_cols, axis=1)

        # splitting on train/val
        X_train, X_val, y_train, y_val = train_test_split(
            feats_df, target_df, test_size=0.2, random_state=42
        )

        # loading image embeddings
        train_images = pd.read_parquet(train_image_path, engine='pyarrow')
        val_images = train_images.loc[train_images.item_id.isin(X_val.item_id)].copy()
        train_images = train_images.loc[train_images.item_id.isin(X_train.item_id)].copy()

        # getting text embeddings
        train_texts = self._prepare_text_features(X_train, True)
        val_texts = self._prepare_text_features(X_val, False)

        # OneHot
        self.condition_encoder = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop=None, sparse_output=False), ['item_condition'])
            ],
            remainder='passthrough'
        )

        X_train_with_onehot = self.condition_encoder.fit_transform(X_train)
        X_val_with_onehot = self.condition_encoder.transform(X_val)

        # getting names of columns
        onehot_feature_names = self.condition_encoder.named_transformers_['onehot'].get_feature_names_out(['item_condition'])
        passthrough_cols = [col for col in X_train.columns if col != 'item_condition']
        all_columns = list(onehot_feature_names) + passthrough_cols

        X_train_with_onehot = pd.DataFrame(
            X_train_with_onehot,
            columns=all_columns,
            index=X_train.index
        )
        X_val_with_onehot = pd.DataFrame(
            X_val_with_onehot,
            columns=all_columns,
            index=X_val.index
        )

        # deleting cols
        self.cols_to_drop = ['image_name', 'title', 'description', 'buyer_id', 'seller_id']

        X_train_with_onehot = X_train_with_onehot.drop(self.cols_to_drop, axis=1)
        X_val_with_onehot = X_val_with_onehot.drop(self.cols_to_drop, axis=1)

        # Target encoding
        self.multiple_target_encoder = MultiTargetEncoder(
            cat_cols=self.cat_cols,
            target_cols=self.target_cols,
            encoder_cls=TargetEncoder,
            encoder_params=dict(
                cv=5,
                target_type='continuous',
                shuffle=True,
                random_state=42
            )
        )

        self.multiple_target_encoder.fit(X_train_with_onehot, y_train)
        X_train_final = self.multiple_target_encoder.transform(X_train_with_onehot)
        X_val_final = self.multiple_target_encoder.transform(X_val_with_onehot)

        train_texts_df = pd.DataFrame(
            train_texts,
            index=X_train_final.index,
            columns=[f'text_emb_{i}' for i in range(train_texts.shape[1])]
        )

        val_texts_df = pd.DataFrame(
            val_texts,
            index=X_val_final.index,
            columns=[f'text_emb_{i}' for i in range(val_texts.shape[1])]
        )

        X_train_final = pd.concat([X_train_final, train_texts_df], axis=1)
        X_val_final = pd.concat([X_val_final, val_texts_df], axis=1)
        return (
            X_train_final.merge(train_images, on='item_id', how='left'),
            X_val_final.merge(val_images, on='item_id', how='left'),
            y_train, y_val
        )

    def transform(self, df_test, test_image_path='image_embeddings/test_embeddings.parquet'):
        """transform test data"""

        if self.condition_encoder is None or self.multiple_target_encoder is None:
            raise ValueError('You need to fit the model before transforming')

        # basic preprocessing
        df_test = self._basic_preprocessing(df_test, False)

        # loading image embeddings
        test_images = pd.read_parquet(test_image_path, engine='pyarrow')
        test_images = test_images.drop('item_id', axis=1, inplace=True)

        # getting text features
        test_texts = self._prepare_text_features(df_test, False)

        # OneHot encoding
        X_test_with_onehot = self.condition_encoder.transform(df_test)

        # getting column names
        onehot_feature_names = self.condition_encoder.named_transformers_['onehot'].get_feature_names_out(['item_condition'])
        passthrough_cols = [col for col in df_test.columns if col != 'item_condition']
        all_columns = list(onehot_feature_names) + passthrough_cols

        X_test_with_onehot = pd.DataFrame(
            X_test_with_onehot,
            columns=all_columns,
            index=df_test.index
        )

        # dropping columns
        X_test_with_onehot = X_test_with_onehot.drop(self.cols_to_drop, axis=1)

        # Target encoding
        X_test_final = self.multiple_target_encoder.transform(X_test_with_onehot)

        test_texts_df = pd.DataFrame(
            test_texts,
            index=X_test_final.index,
            columns=[f'text_emb_{i}' for i in range(test_texts.shape[1])]
        )
        # final concat
        X_test_complete = pd.concat([X_test_final, test_texts_df], axis=1)
        X_test_complete = pd.concat([X_test_complete, test_images], axis=1)
        return X_test_complete
