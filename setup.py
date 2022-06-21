from setuptools import find_packages, setup

setup(
    name='classification_wrapper',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'scikit-learn==0.21.2',
        'pandas==0.24.2',
        'numpy==1.22.0'
    ],
)