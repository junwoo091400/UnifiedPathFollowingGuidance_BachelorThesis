from setuptools import setup
setup(    
    name="windywings-gym",
    version='0.0.1',
    packages=["windywings"],
    install_requires=['gym[classic_control]',
                      'numpy',
                      'matplotlib']
)
