# -*- coding: utf-8 -*-

import tf_quat2rot
import tensorflow as tf
tf.enable_eager_execution()


class TestConversion(tf.test.TestCase):

    def test_unit_rotation_float32(self):

        unit_quaternion = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        rotation_matrix = tf_quat2rot.quaternion_to_rotation_matrix(unit_quaternion)
        self.assertAllClose(tf.eye(3, dtype=unit_quaternion.dtype), rotation_matrix)

        identity_rotation_matrix = tf.eye(3, dtype=tf.float32)
        quaternion = tf_quat2rot.rotation_matrix_to_quaternion(identity_rotation_matrix)
        self.assertAllClose(tf.constant([1.0, 0.0, 0.0, 0.0], dtype=identity_rotation_matrix.dtype),
                            quaternion)

    def test_unit_rotation_float64(self):

        unit_quaternion = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float64)
        rotation_matrix = tf_quat2rot.quaternion_to_rotation_matrix(unit_quaternion)
        self.assertAllClose(tf.eye(3, dtype=unit_quaternion.dtype), rotation_matrix)

        identity_rotation_matrix = tf.eye(3, dtype=tf.float64)
        quaternion = tf_quat2rot.rotation_matrix_to_quaternion(identity_rotation_matrix)
        self.assertAllClose(tf.constant([1.0, 0.0, 0.0, 0.0], dtype=identity_rotation_matrix.dtype),
                            quaternion)

    def test_xy_flip(self):
        rotation_matrix = tf.constant(
            [[-1.0,  0.0, 0.0],
             [ 0.0, -1.0, 0.0],
             [ 0.0,  0.0, 1.0]], dtype=tf.float32)
        quaternion = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrix)
        self.assertAllClose(tf.constant([0.0, 0.0, 0.0, 1.0], dtype=rotation_matrix.dtype),
                            tf.abs(quaternion))


if __name__ == '__main__':
    tf.test.main()  # run all unit tests﻿
