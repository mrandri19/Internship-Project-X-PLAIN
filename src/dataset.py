from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset:
    @classmethod
    def from_indices(cls, indices: [int], other):
        return cls(other._decoded_df.copy().iloc[indices], other.columns)

    def __init__(self, data, attributes):
        self._decoded_df = pd.DataFrame(data)
        self.columns = attributes

        # Rename columns from 0,1,... to the attributes[0,1,...][0]
        columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in attributes])}
        self._decoded_df = self._decoded_df.rename(columns=columns_mapper)

        dict_columns = {k: v for (k, v) in self.columns}

        # Encode categorical columns with value between 0 and n_classes-1
        # Keep the columns encoders used to perform the inverse transformation
        # https://stackoverflow.com/a/31939145
        def func(x):
            self._column_encoders[x.name].fit(dict_columns[x.name])
            return self._column_encoders[x.name].transform(x)

        self._column_encoders = defaultdict(LabelEncoder)
        self._encoded_df = self._decoded_df.apply(func)

    def class_values(self):
        """All possible classes in the dataset"""
        return self.columns[-1][1]

    def X(self):
        """All rows' attributes as a pandas DataFrame."""
        return self._encoded_df.iloc[:, :-1]

    def Y_decoded(self):
        """All rows' classes as a pandas Series."""
        return self._decoded_df.iloc[:, -1]

    def X_numpy(self):
        """All rows' attributes as a numpy float64 array."""
        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def Y_numpy(self):
        """All rows' classes as a numpy float64 array."""
        return self._encoded_df.iloc[:, -1].to_numpy().astype(np.float64)

    def attributes(self):
        return self.columns[:-1]

    def class_column_name(self):
        """"The column name of the class attribute"""
        return self.columns[-1][0]

    def __len__(self):
        return len(self._decoded_df)

    def __getitem__(self, item) -> pd.Series:
        """Returns the i-th element of the encoded DataFrame of datset"""
        return self._encoded_df.iloc[item]

    def get_decoded(self, item) -> pd.Series:
        """Returns the i-th element of the decoded DataFrame of datset"""
        return self._decoded_df.iloc[item]

    def transform_instance(self, decoded_instance: pd.Series) -> pd.Series:
        """Transform a decoded instance to an encoded instance using the Dataset's column encoders"""
        return pd.Series(
            {col: self._column_encoders[col].transform([val])[0]
             for (col, val)
             in
             decoded_instance.items()}
        )

    def inverse_transform_instance(self, encoded_instance: pd.Series) -> pd.Series:
        return pd.Series(
            {col: self._column_encoders[col].inverse_transform([val])[0]
             for (col, val)
             in encoded_instance.items()}
        )

    def to_arff_obj(self) -> object:
        obj = {'relation': self.class_column_name(),
               'attributes': self.columns,
               'data': self._decoded_df.values.tolist()}
        return obj
