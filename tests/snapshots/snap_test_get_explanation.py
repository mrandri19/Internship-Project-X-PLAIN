# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import GenericRepr, Snapshot


snapshots = Snapshot()

snapshots['TestGet_explanation::test_get_explanation_adult_naive_bayes 1'] = (
    GenericRepr('<src.XPLAIN_explainer.XPLAIN_explainer object at 0x100000000>'),
    [
        0.04267021431143747,
        0.0013601660256593595,
        -0.00025793916220828716,
        -0.002583202995719347,
        0.41800765869397466,
        -0.07252560975447497,
        0.038882713120766854,
        -0.0002026726258052003,
        6.138967731539324e-05,
        0.01974576342296308,
        -0.013082534026229053
    ],
    {
        '1,10,11': 0.04904249942186634,
        '1,3,5,6,8,9,10,11': 0.32111161850226366,
        '1,3,8,9': 0.04214440558178556,
        '1,6': -0.044924825486333586,
        '5': 0.41800765869397466
    },
    118,
    0.01442188127852051,
    GenericRepr('sex                      Female\nworkclass               Private\neducation               Dropout\nrace                      White\nmarital-status        Separated\noccupation              Service\nrelationship      Not-in-family\ncapital-gain                low\ncapital-loss                low\nhours-per-week        39.5-40.5\nage                        >=51\nclass                     <=50K\ndtype: object'),
    '<=50K',
    0,
    0.8066897372237432
)

snapshots['TestGet_explanation::test_get_explanation_zoo_naive_bayes 1'] = (
    GenericRepr('<src.XPLAIN_explainer.XPLAIN_explainer object at 0x100000000>'),
    [
        0.0018493244327110192,
        0.0005835022761034869,
        0.04128880084184605,
        0.002583590538021041,
        -0.002213184871231788,
        -0.0005848709570587252,
        -0.0026459469615073283,
        -0.0027706378431165968,
        -0.0033543844546455315,
        -0.0033771583400726835,
        -0.0023206657800494135,
        -0.0032075223689916887,
        -0.0027332607152817934,
        -0.003227422277876868,
        -0.0035048571858332656,
        -0.0015305424415449354
    ],
    {
        '2,3,4,8,9,10,11': 0.39514075431687523
    },
    42,
    0.19470029326431426,
    GenericRepr('hair             1\nfeathers         0\neggs             0\nmilk             1\nairborne         0\naquatic          0\npredator         0\ntoothed          1\nbackbone         1\nbreathes         1\nvenomous         0\nfins             0\nlegs             4\ntail             1\ndomestic         0\ncatsize          1\ntype        mammal\ndtype: object'),
    'mammal',
    5,
    0.9972484549885969
)

snapshots['TestGet_explanation::test_get_explanation_zoo_random_forest 1'] = (
    GenericRepr('<src.XPLAIN_explainer.XPLAIN_explainer object at 0x100000000>'),
    [
        0.11024691358024685,
        0.02308641975308645,
        0.19728395061728388,
        0.31407407407407417,
        0.004938271604938316,
        0.006913580246913575,
        0.0,
        0.07111111111111112,
        0.00864197530864197,
        0.03358024691358019,
        0.0007407407407408195,
        0.0,
        0.005185185185185182,
        0.0,
        0.0,
        0.0
    ],
    {
        '1,2,3,4,8,9,10,11': 0.5839506172839506
    },
    33,
    0.00864197530864197,
    GenericRepr('hair             1\nfeathers         0\neggs             0\nmilk             1\nairborne         0\naquatic          0\npredator         0\ntoothed          1\nbackbone         1\nbreathes         1\nvenomous         0\nfins             0\nlegs             4\ntail             1\ndomestic         0\ncatsize          1\ntype        mammal\ndtype: object'),
    'mammal',
    5,
    1.0
)
