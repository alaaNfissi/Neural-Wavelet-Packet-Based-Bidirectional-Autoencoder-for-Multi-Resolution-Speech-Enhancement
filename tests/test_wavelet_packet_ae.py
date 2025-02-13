import unittest
import torch
from src.models.wavelet_packet_ae import WaveletPacketAutoencoder
import pywt

class TestWaveletPacketAE(unittest.TestCase):
    def setUp(self):
        # Use a small example: 1D tensor of length 256.
        self.input = torch.randn(1, 1, 256)
        # Use Daubechies 20 for kernel initialization.
        wv = pywt.Wavelet("db20")
        self.kernelInit = torch.tensor(pywt.Wavelet("db20").filter_bank[0], dtype=torch.float32)
        self.model = WaveletPacketAutoencoder(stride=2, kernelInit=self.kernelInit, kernTrainable=True,
                                               level=3, kernelsConstraint="Bidirectional",
                                               initThP=1.0, initThN=1.0, trainHT=True, alpha=10, beta=10)
    def test_forward(self):
        reconstructed, detail = self.model(self.input)
        self.assertEqual(reconstructed.shape[2], self.input.shape[2])
        self.assertIsInstance(detail, list)

if __name__ == "__main__":
    unittest.main()
