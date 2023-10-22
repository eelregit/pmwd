from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).resolve().parent
long_description = (this_directory / 'README.rst').read_text()


vis_require = ['matplotlib', 'scipy']
#docs_require = ['sphinx', 'jupyterlab']
#tests_require = ['pytest', 'pytest-cov', 'pytest-benchmark', 'pytest-xdist', 'scipy']


setup(
    name='pmwd',
    description='particle mesh with derivatives',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/eelregit/pmwd',
    author='Yin Li and pmwd developers',
    author_email='eelregit@gmail.com',
    use_scm_version={'write_to': 'pmwd/_version.py'},
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'jax>=0.4.7',
        'mcfit>=0.0.18',  # jax backend
    ],
    extras_require={
        'vis': vis_require,
        #'docs': docs_require,
        #'tests': tests_require,
        #'dev': vis_require + docs_require + tests_require,
    }
)
