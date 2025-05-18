from setuptools import setup, find_packages
setup(
    name="FFNet",
    version="0.1.0",
    packages=find_packages(),  # 自动发现所有包
    install_requires=[         # 依赖库
        "numpy>=1.20.0",
        "torch>=2.4.0",
    ],
    python_requires=">=3.9",   # Python版本要求
)


