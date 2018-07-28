from setuptools import setup, find_packages

setup(
    name='mlragtag',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A rag tag collection of code snippets to be able to import to Kaggle',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/rh314/mlragtag',
    author='Ruan Havenstein',
    author_email='rh314@users.noreply.github.com'
)
