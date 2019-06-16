# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = 'c.rist@posteo.de'

import tensorflow as tf


def assert_normalized_quaternion(quaternion: tf.Tensor):
    with tf.control_dependencies([
        tf.debugging.assert_near(
            tf.ones_like(quaternion[..., 0]), tf.linalg.norm(quaternion, axis=-1),
            message='Input quaternions are not normalized.')]):
        return quaternion


def assert_valid_rotation(rotation_matrix: tf.Tensor):
    r = rotation_matrix
    with tf.control_dependencies([
        tf.debugging.assert_near(tf.ones_like(rotation_matrix[..., 0, 0]),
                                 tf.linalg.det(rotation_matrix),
                                 message="Invalid rotation matrix."),
        tf.debugging.assert_near(tf.linalg.matmul(tf.linalg.transpose(r), r),
                                 tf.eye(3, batch_shape=tf.shape(r)[:-2], dtype=r.dtype),
                                 message="Invalid rotation matrix.")
    ]):
        return r
