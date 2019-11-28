from snapshottest import TestCase
from XPLAIN_utils.LACE_utils.LACE_utils1 import loadARFF_Weka


class TestLoadARFF_Weka(TestCase):
    def test_loadARFF_Weka(self):
        dataset = loadARFF_Weka("datasets/adult_d.arff")
        self.assertMatchSnapshot((
            dataset.X,
            dataset.Y,
            dataset.W,
            dataset.metas
        ))
