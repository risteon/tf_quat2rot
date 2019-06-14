# tf_quat2rot
Convert between quaternion and rotation matrices in tensorflow.


Implements the method from Soheil Sarabandi and Federico Thomas to compute rotation matrices
from quaternions.
This method is numerically stable.

Written in pure tensorflow.

## Definitions

Quaternions are defined as ```w-x-y-z```. ```w``` is defined as positive.

## Reference
Sarabandi, S., & Thomas, F. (2019). Accurate Computation of Quaternions from Rotation Matrices. Advances in Robot Kinematics 2018, 39–46. https://doi.org/10.1007/978-3-319-93188-3_5
