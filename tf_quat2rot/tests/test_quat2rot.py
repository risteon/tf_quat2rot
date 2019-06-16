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

    def test_flips(self):
        rotation_matrices = tf.constant(
            [
                [
                    [-1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0,  1.0]
                ],
                [
                    [ 1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0, -1.0]
                ],
                [
                    [-1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0],
                    [ 0.0,  0.0, -1.0]
                ]
            ],
            dtype=tf.float64
        )
        quaternions = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrices,
                                                                assert_valid=True)
        self.assertAllClose(tf.constant([[0.0, 0.0, 0.0, 1.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 0.0]], dtype=rotation_matrices.dtype),
                            tf.abs(quaternions))

    def test_unnormalized_quaternion(self):
        unit_quaternion = tf.constant([0.99, 0.0, 0.0, 0.0], dtype=tf.float64)
        _ = tf_quat2rot.quaternion_to_rotation_matrix(unit_quaternion, assert_normalized=False)
        with self.assertRaises(tf.errors.InvalidArgumentError) as context:
            _ = tf_quat2rot.quaternion_to_rotation_matrix(unit_quaternion, assert_normalized=True)

        self.assertTrue('not normalized' in str(context.exception))
        r = tf_quat2rot.quaternion_to_rotation_matrix(unit_quaternion, assert_normalized=True,
                                                      normalize=True)
        self.assertAllClose(tf.eye(3, dtype=unit_quaternion.dtype), r)

    def test_invalid_rotation(self):
        rotation_matrix = tf.constant(
            [[-1.0, 0.001, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0]], dtype=tf.float64)
        _ = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrix)
        with self.assertRaises(tf.errors.InvalidArgumentError) as context:
            _ = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrix,
                                                          assert_valid=True)

        self.assertTrue('Invalid rotation matrix.' in str(context.exception))


class TestRandomRotations(tf.test.TestCase):

    def test_generation_smoke(self):
        # single random quaternion
        _ = tf_quat2rot.random_uniform_quaternion(assert_normalized=True)
        # multiple random quaternions
        q = tf_quat2rot.random_uniform_quaternion(batch_dim=(2, 3), assert_normalized=True)
        self.assertEqual((2, 3, 4), q.shape)
        # random batch dimension (==dynamic batch dimension)
        bd = tf.random.uniform(shape=(3, ), dtype=tf.int32, minval=1, maxval=5)
        _ = tf_quat2rot.random_uniform_quaternion(batch_dim=bd, assert_normalized=True)

    def test_cycle_conversion_qrq(self):
        random_quats = tf_quat2rot.random_uniform_quaternion(batch_dim=(3, 4, 5))
        random_rotations = tf_quat2rot.quaternion_to_rotation_matrix(random_quats)
        random_quats_restored = tf_quat2rot.rotation_matrix_to_quaternion(random_rotations)
        self.assertEqual((3, 4, 5, 4), random_quats_restored.shape)
        self.assertAllClose(random_quats, random_quats_restored)

    def test_cycle_conversion_rqr(self):
        random_rot = tf_quat2rot.random_uniform_rotation_matrix(batch_dim=(2, 5),
                                                                assert_valid=True)
        random_quat = tf_quat2rot.rotation_matrix_to_quaternion(random_rot)
        random_rot_restored = tf_quat2rot.quaternion_to_rotation_matrix(random_quat)
        self.assertEqual((2, 5, 3, 3), random_rot_restored.shape)
        self.assertAllClose(random_rot, random_rot_restored)


if __name__ == '__main__':
    tf.test.main()  # run all unit tests﻿
