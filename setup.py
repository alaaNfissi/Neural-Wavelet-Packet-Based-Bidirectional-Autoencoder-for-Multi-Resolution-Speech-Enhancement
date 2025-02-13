from setuptools import find_packages, setup

setup(
    name="wavelet_packet_ae",
    version="0.1.0",
    description="Neural Wavelet Packet-Based Bidirectional Autoencoder for Multi-Resolution Speech Enhancement",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "numpy",
        "pandas",
        "matplotlib",
        "pywt",
        "pysepm",
        "pystoi",
        "pesq",
        "scikit-learn",
        "tqdm",
        "tensorboard",
        "ray[tune]",
    ],
    entry_points={
        "console_scripts": [
            "run_wavelet_ae=main:main",
        ],
    },
)
