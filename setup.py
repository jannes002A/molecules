from setuptools import setup

setup(
    name='test_molecules',
    version='0.1.0',
    description='A test package',
    install_requires=['gym',
                      'jax',
                      'pandas',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'torch',
                      'sklearn',
                     ],

    classifiers=[
        'Development Status :: 1-Testing'
    ],
)
