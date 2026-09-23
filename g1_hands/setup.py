from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['g1_hands'],
    package_dir={'g1_hands': 'src/g1_hands'},
)

setup(**d)
