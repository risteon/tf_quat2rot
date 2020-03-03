# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = "c.rist@posteo.de"

import tf_quat2rot
import tensorflow as tf

# flag if tensorflow 1.X is in use (no eager execution)
tf_version_1 = int(tf.__version__.split(".")[0]) < 2


class TestConversion(tf.test.TestCase):
    def test_identity_rotation(self, dtype=tf.dtypes.float32):
        with self.session(use_gpu=False):
            identity_quaternion = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=dtype)
            identity_rotation_matrix = tf.eye(3, dtype=identity_quaternion.dtype)
            rotation_matrix = tf_quat2rot.quaternion_to_rotation_matrix(
                identity_quaternion
            )
            quaternion = tf_quat2rot.rotation_matrix_to_quaternion(
                identity_rotation_matrix
            )
            self.assertAllClose(identity_rotation_matrix, rotation_matrix)
            self.assertAllClose(identity_quaternion, quaternion)

    def test_identity_rotation_float64(self):
        self.test_identity_rotation(dtype=tf.dtypes.float64)

    def test_xy_flip(self, dtype=tf.dtypes.float32):
        with self.session(use_gpu=False):
            rotation_matrix = tf.constant(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
            )
            quaternion = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrix)
            quaternion = tf.abs(quaternion)
            expected = tf.constant([0.0, 0.0, 0.0, 1.0], dtype=rotation_matrix.dtype)
            self.assertAllClose(expected, quaternion)

    def test_xy_flip_float64(self):
        self.test_xy_flip(dtype=tf.dtypes.float64)

    def test_flips(self):
        with self.session(use_gpu=False):
            rotation_matrices = tf.constant(
                [
                    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                ],
                dtype=tf.float64,
            )
            quaternions = tf_quat2rot.rotation_matrix_to_quaternion(
                rotation_matrices, assert_valid=True
            )
            quaternions = tf.abs(quaternions)
            expected = tf.constant(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=rotation_matrices.dtype,
            )

            self.assertAllClose(expected, quaternions)

    def test_unnormalized_quaternion(self, dtype=tf.dtypes.float32):
        with self.session(use_gpu=False):
            scaled_identity_quaternion = tf.constant([0.99, 0.0, 0.0, 0.0], dtype=dtype)

            _ = tf_quat2rot.quaternion_to_rotation_matrix(
                scaled_identity_quaternion, assert_normalized=False
            )
            if tf_version_1:
                _.eval()

            with self.assertRaises(tf.errors.InvalidArgumentError) as context:
                _ = tf_quat2rot.quaternion_to_rotation_matrix(
                    scaled_identity_quaternion, assert_normalized=True
                )
                if tf_version_1:
                    _.eval()

            self.assertTrue("not normalized" in str(context.exception))
            r = tf_quat2rot.quaternion_to_rotation_matrix(
                scaled_identity_quaternion, assert_normalized=True, normalize=True
            )
            expected = tf.eye(3, dtype=scaled_identity_quaternion.dtype)
            self.assertAllClose(expected, r)

    def test_unnormalized_quaternion_float64(self):
        self.test_unnormalized_quaternion(dtype=tf.dtypes.float64)

    def test_invalid_rotation(self):
        with self.session(use_gpu=False):
            rotation_matrix = tf.constant(
                [[-1.0, 0.001, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=tf.float64,
            )
            _ = tf_quat2rot.rotation_matrix_to_quaternion(rotation_matrix)
            if tf_version_1:
                _.eval()
            with self.assertRaises(tf.errors.InvalidArgumentError) as context:
                _ = tf_quat2rot.rotation_matrix_to_quaternion(
                    rotation_matrix, assert_valid=True
                )
                if tf_version_1:
                    _.eval()

            self.assertTrue("Invalid rotation matrix." in str(context.exception))


class TestRandomRotations(tf.test.TestCase):
    def test_generation_smoke(self):
        with self.session(use_gpu=False):
            # single random quaternion
            _ = tf_quat2rot.random_uniform_quaternion(assert_normalized=True)
            if tf_version_1:
                _.eval()
            # multiple random quaternions
            q = tf_quat2rot.random_uniform_quaternion(
                batch_dim=(2, 3), assert_normalized=True
            )
            self.assertEqual((2, 3, 4), q.shape)
            # random batch dimension (==dynamic batch dimension)
            bd = tf.random.uniform(shape=(3,), dtype=tf.int32, minval=1, maxval=5)
            _ = tf_quat2rot.random_uniform_quaternion(
                batch_dim=bd, assert_normalized=True
            )
            if tf_version_1:
                _.eval()

    def test_cycle_conversion_qrq(self):
        with self.session(use_gpu=False) as sess:
            batch_shape = (3, 4, 5)
            random_quats = tf_quat2rot.random_uniform_quaternion(batch_dim=batch_shape)

            random_rotations = tf_quat2rot.quaternion_to_rotation_matrix(random_quats)
            random_quats_restored = tf_quat2rot.rotation_matrix_to_quaternion(
                random_rotations
            )
            self.assertEqual(batch_shape + (4,), random_quats_restored.shape)

            if tf_version_1:
                random_quats, random_quats_restored = sess.run(
                    [random_quats, random_quats_restored]
                )
            self.assertAllClose(random_quats, random_quats_restored)

    def test_cycle_conversion_rqr(self):
        with self.session(use_gpu=False) as sess:
            batch_shape = (2, 5)
            random_rots = tf_quat2rot.random_uniform_rotation_matrix(
                batch_dim=batch_shape, assert_valid=True
            )
            random_quats = tf_quat2rot.rotation_matrix_to_quaternion(random_rots)
            random_rot_restored = tf_quat2rot.quaternion_to_rotation_matrix(
                random_quats
            )
            self.assertEqual(batch_shape + (3, 3), random_rot_restored.shape)

            if tf_version_1:
                random_rots, random_rot_restored = sess.run(
                    [random_rots, random_rot_restored]
                )

            self.assertAllClose(random_rots, random_rot_restored)


if __name__ == "__main__":
    tf.test.main()  # run all unit tests﻿
