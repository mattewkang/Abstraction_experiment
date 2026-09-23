from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['g1_arm_abs'],
    package_dir={'g1_arm_abs': 'src/g1_arm_abs'},
)

setup(**d)
