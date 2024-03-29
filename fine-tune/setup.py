from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='EmbeddingFineTune',
    version='1.0',
    description='EmbeddingFineTune',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='jvamaraju@nvidia.com',
    url='https://github.com/FlagOpen/FlagEmbedding',
    packages=find_packages(),
    install_requires=[
        'torch>=1.6.0',
        'transformers==4.34.0',
        'datasets',
        'accelerate>=0.20.1',
        'sentence_transformers',
    ],
)