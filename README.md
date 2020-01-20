# tf_quat2rot
[![Build Status](https://travis-ci.org/risteon/tf_quat2rot.svg?branch=master)](https://travis-ci.org/risteon/tf_quat2rot)
[![License](https://img.shields.io/pypi/l/tf-quat2rot)](https://pypi.org/project/tf-quat2rot/)
[![PyPI version](https://img.shields.io/pypi/v/tf-quat2rot)](https://pypi.org/project/tf-quat2rot/)

Convert between quaternion and rotation matrices in tensorflow.


Implements the method from Soheil Sarabandi and Federico Thomas to compute rotation matrices
from quaternions.
This method is numerically stable.

Written in pure tensorflow.

## Installation

Python Versions from 3.5+ are supported. Install from PyPi with `pip install tf_quat2rot`.
`tf_quat2rot` requires TensorFlow (versions 1.x and 2.x work). If you want it to be installed when installing `tf_quat2rot` choose between the CPU and GPU version with `pip install tf_quat2rot[tf-cpu]` or `pip install tf_quat2rot[tf-gpu]` respectively.

## Definitions and sample usage

Quaternions are defined as `w-x-y-z`. `w` is defined as positive. Usage is straightforward:

```python3
>>> import tensorflow as tf
>>> if int(tf.__version__.split('.')[0]) < 2:
>>>     tf.enable_eager_execution()

>>> from tf_quat2rot import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

>>> r = quaternion_to_rotation_matrix(tf.constant([1.0, 0.0, 0.0, 0.0]))
>>> print(r)
tf.Tensor(
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)

>>> q = rotation_matrix_to_quaternion(tf.eye(3))
>>> print(q)
tf.Tensor([ 1. -0. -0. -0.], shape=(4,), dtype=float32)
```

Quaternion and rotation matrix tensors can hold arbitrary leading dimensions which will
be preserved.

Use `random_uniform_quaternion` to generate quaternions uniformly from SO(3).

## References
Sarabandi, S., & Thomas, F. (2019). Accurate Computation of Quaternions from Rotation Matrices.
Advances in Robot Kinematics 2018, 39–46. https://doi.org/10.1007/978-3-319-93188-3_5

Steven M LaValle. Generating a random element of SO(3). http://planning.cs.uiuc.edu/node198.html
