"""Setup configuration for v2t-service package."""

from setuptools import setup, find_packages

setup(
    name="v2t-service",
    version="0.1.0",
    description="Voice-to-Text service with ASR and VAD support",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.21.0",
        "faster-whisper>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
    },
)

