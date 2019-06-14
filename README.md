# tf_quat2rot
Convert between quaternion and rotation matrices in tensorflow.


Implements the method from Soheil Sarabandi and Federico Thomas to compute rotation matrices
from quaternions.
This method is numerically stable.

Written in pure tensorflow.

## Installation

From PyPi with ```pip install tf_quat2rot```

## Definitions and sample usage

Quaternions are defined as ```w-x-y-z```. ```w``` is defined as positive. Usage is straightforward:

```python3
import tensorflow as tf
from tf_quat2rot import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

rotation_matrix = quaternion_to_rotation_matrix(tf.constant([1.0, 0.0, 0.0, 0.0]))
quaternion = rotation_matrix_to_quaternion(tf.eye(3))
```

Quaternion and rotation matrix tensors can hold arbitrary leading dimensions which will
be preserved.

## Reference
Sarabandi, S., & Thomas, F. (2019). Accurate Computation of Quaternions from Rotation Matrices.
Advances in Robot Kinematics 2018, 39–46. https://doi.org/10.1007/978-3-319-93188-3_5
