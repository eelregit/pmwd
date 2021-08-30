from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).resolve().parent
long_description = (this_directory / 'README.rst').read_text()


setup(
    name='pmwd',
    description='particle mesh with derivatives',
    long_description=long_description,
    long_description_content_type='text/x-rst'
    url='https://github.com/eelregit/pmwd',
    author='Yin Li and pmwd developers',
    author_email='eelregit@gmail.com',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['jax'],
)
