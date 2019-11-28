# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import GenericRepr, Snapshot


snapshots = Snapshot()

snapshots['TestLoadARFF_Weka::test_loadARFF_Weka 1'] = (
    GenericRepr('array([[0., 2., 1., ..., 1., 2., 0.],\n       [1., 2., 0., ..., 1., 0., 0.],\n       [1., 1., 4., ..., 1., 1., 1.],\n       ...,\n       [1., 4., 4., ..., 1., 1., 2.],\n       [1., 5., 1., ..., 1., 3., 1.],\n       [1., 4., 2., ..., 1., 1., 1.]])'),
    GenericRepr('array([0., 0., 0., ..., 1., 1., 1.])'),
    GenericRepr('array([], shape=(14000, 0), dtype=float64)'),
    GenericRepr('array([], shape=(14000, 0), dtype=object)')
)
