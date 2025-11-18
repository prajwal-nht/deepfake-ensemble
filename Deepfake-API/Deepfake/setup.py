from setuptools import setup, find_packages

setup(
    name="deepfake-detection-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "opencv-python-headless>=4.5.3",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pillow>=8.3.1",
        "scikit-learn>=0.24.2",
    ],
    python_requires=">=3.8",
)
