from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='MetaArray',
    version='0.1.0',
    author='Matthew R. Argall',
    author_email='matthew.argall@eunh.edu',
    description='Self-documenting and plotting arrays.',
    long_description=long_description,
    long_description_context_type='text/markdown',
    url='https://github.com/argallmr/pyarray',
    license='MIT'
    keywords='array, multidimensional arrays',
    packages=['metaarray'],
    license='LICENSE.txt',
    install_requires=[
                      ],
    python_requires='>=3.6'
    )