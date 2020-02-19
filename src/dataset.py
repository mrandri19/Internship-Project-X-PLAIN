from collections import defaultdict

import Orange
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, data, attributes):
        self._decoded_df = pd.DataFrame(data)
        self.columns = attributes

        # Rename columns from 0,1,... to the attributes[0,1,...][0]
        columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in attributes])}
        self._decoded_df = self._decoded_df.rename(columns=columns_mapper)

        # Encode categorical columns with value between 0 and n_classes-1
        # Keep the columns encoders used to perform the inverse transformation
        # https://stackoverflow.com/a/31939145
        self._column_encoders = defaultdict(LabelEncoder)
        self._encoded_df = self._decoded_df.apply(
            lambda x: self._column_encoders[x.name].fit_transform(x))

    def class_values(self):
        """All possible classes in the dataset"""
        return self.columns[-1][1]

    def X(self):
        """All rows' attributes as a pandas DataFrame."""
        return self._encoded_df.iloc[:, :-1]

    def Y(self):
        """All rows' classes as a pandas Series."""
        return self._encoded_df.iloc[:, -1]

    def X_numpy(self):
        """All rows' attributes as a numpy float64 array."""
        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def Y_numpy(self):
        """All rows' classes as a numpy float64 array."""
        return self._encoded_df.iloc[:, -1].to_numpy().astype(np.float64)

    def attributes(self):
        return self.columns[:-1]

    def row_inverse_transform_value(self, value, column):
        """"Given a value (one column of a row) and that column's name, return itsdecoded value"""
        return self._column_encoders[column].inverse_transform(value)

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

    def orange_domain(self) -> Orange.data.Domain:
        """"Return a Orange.data.Domain built using the dataset's attributes"""
        orange_vars = [Orange.data.DiscreteVariable.make(name, vals) for (name, vals) in
                       self.columns]
        return Orange.data.Domain(attributes=orange_vars[:-1], class_vars=orange_vars[-1])

    def to_arff_obj(self) -> object:
        obj = {'relation': self.class_column_name(),
               'attributes': self.columns,
               'data': self._decoded_df.values.tolist()}
        return obj

    def to_orange_table(self) -> Orange.data.Table:
        return Orange.data.Table.from_numpy(self.orange_domain(), self.X_numpy(),
                                            self.Y_numpy())
