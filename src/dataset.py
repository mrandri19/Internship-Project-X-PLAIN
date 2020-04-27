"""This module provides the Dataset class, which is used to describe your
dataset in a format suitable for the $name analysis.

"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset:
    """The Dataset class contains information on a dataset and all of the
    possible values of every one of each columns.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Anything that can be converted to a pandas DataFrame.

        This data, after being converted into a DataFrame, has every row processed by
        a scikit-learn LabelEncoder.

        An example:
        ::
            [
                ['1','0','0','1','0','0','1','1','1','1','0','0','4','0','0','1','mammal'],
                ['1','0','0','1','0','0','0','1','1','1','0','0','4','1','0','1','mammal']
            ]
    columns : list
        A list of tuples. Each tuple corresponds to one of the columns of `data`
        and contains its name and a list of **all** its possible values.

        An example (all of the examples will use the UCI ML Zoo dataset):
        ::

            [
                ('hair', ['0', '1']),
                ('feathers', ['0', '1']),
                ('eggs', ['0', '1']),
                ('milk', ['0', '1']),
                ('airborne', ['0', '1']),
                ('aquatic', ['0', '1']),
                ('predator', ['0', '1']),
                ('toothed', ['0', '1']),
                ('backbone', ['0', '1']),
                ('breathes', ['0', '1']),
                ('venomous', ['0', '1']),
                ('fins', ['0', '1']),
                ('legs', ['0', '2', '4', '5', '6', '8']),
                ('tail', ['0', '1']),
                ('domestic', ['0', '1']),
                ('catsize', ['0', '1']),
                ('type', ['amphibian','bird','fish','insect','invertebrate','mammal','reptile'])
            ]

    Attributes
    ----------
    columns : list
        The same `columns` object passed in the constructor


    """

    @classmethod
    def from_indices(cls, indices: [int], other):
        return cls(other._decoded_df.copy().iloc[indices], other.columns)

    def __init__(self, data, columns):
        self._decoded_df = pd.DataFrame(data)
        self.columns = columns

        # Rename columns from 0,1,... to the attributes[0,1,...][0]
        columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in columns])}
        self._decoded_df = self._decoded_df.rename(columns=columns_mapper)

        dict_columns = {k: v for (k, v) in self.columns}

        # TODO(Andrea): Use new OrdinalEncoder here

        # Encode categorical columns with value between 0 and n_classes-1
        # Keep the columns encoders used to perform the inverse transformation
        # https://stackoverflow.com/a/31939145
        def func(x):
            self._column_encoders[x.name].fit(dict_columns[x.name])
            return self._column_encoders[x.name].transform(x)

        self._column_encoders = defaultdict(LabelEncoder)
        self._encoded_df = self._decoded_df.apply(func)

    def class_values(self):
        """All the possible classes of an instance of the dataset
        ::

            In[35]: d.class_values()
            Out[35]: ['amphibian', 'bird', 'fish', 'insect', 'invertebrate', 'mammal', 'reptile']
        """
        return self.columns[-1][1]

    def X(self):
        """All rows' attributes as a pandas DataFrame. These attributes were
        encoded with scikit-learn's Label Encoder. See `X_decoded()` to get
        the original data.
        ::

            In[28]: d.X()
            Out[28]:
               hair  feathers  eggs  milk  airborne  ...  fins  legs  tail  domestic  catsize
            0     1         0     0     1         0  ...     0     2     0         0        1
            1     1         0     0     1         0  ...     0     2     1         0        1
        """
        return self._encoded_df.iloc[:, :-1]

    def Y(self):
        """All rows' classes as a pandas DataFrame. these classes were
        encoded with scikit-learn's Label Encoder. See `Y_decoded()` to get
        the original data.
        ::

            In[32]: d.Y()
            Out[32]:
            0    5
            1    5
            Name: type, dtype: int64
        """
        return self._encoded_df.iloc[:, -1]

    def X_decoded(self):
        """All rows' attributes as a pandas DataFrame.
        ::

            d.X_decoded()
            Out[29]:
              hair feathers eggs milk airborne  ... fins legs tail domestic catsize
            0    1        0    0    1        0  ...    0    4    0        0       1
            1    1        0    0    1        0  ...    0    4    1        0       1
        """
        return self._decoded_df.iloc[:, :-1]

    def Y_decoded(self):
        """All rows' classes as a pandas DataFrame.
        ::

            In[33]: d.Y_decoded()
            Out[33]:
            0    mammal
            1    mammal
            Name: type, dtype: object
        """
        return self._decoded_df.iloc[:, -1]

    def X_numpy(self):
        """All encoded rows' attributes as a numpy float64 array.
        ::

            In[30]: d.X_numpy()
            Out[30]:
            array([[1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 2., 0., 0., 1.],
                   [1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 2., 1., 0., 1.]])
        """

        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def Y_numpy(self):
        """All rows' classes as a numpy float64 array.
        ::

            In[34]: d.Y_numpy()
            Out[34]: array([5., 5.])
        """

        return self._encoded_df.iloc[:, -1].to_numpy().astype(np.float64)

    def attributes(self):
        return self.columns[:-1]

    def class_column_name(self):
        """The column name of the class attribute
        ::

            In[36]: d.class_column_name()
            Out[36]: 'type'
        """
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
