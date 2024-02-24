import unittest
import torch
from gateloop_transformer import Transformer

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestGateloopTransformer(unittest.TestCase):

    @staticmethod
    def _create_input(batch_size):
        return torch.randint(0, 256, (batch_size, 1024))
    

    def _test_forward(self, batch_size):
        x = self._create_input(batch_size)
        transformer = Transformer(
            dim = 512,
            num_tokens = 256,
            depth = 3
        )
        _ = transformer(x)


    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)