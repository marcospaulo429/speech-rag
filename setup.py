from setuptools import setup, find_packages

setup(
    name="speech-rag",
    version="0.1.0",
    description="Speech Retrieval-Augmented Generation without Automatic Speech Recognition",
    author="SpeechRAG Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "torchaudio>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
    },
)

