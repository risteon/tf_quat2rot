# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = "c.rist@posteo.de"

import tensorflow as tf
from .check import assert_normalized_quaternion, assert_valid_rotation


def quaternion_to_rotation_matrix(
    quaternion: tf.Tensor, assert_normalized: bool = False, normalize: bool = False
):
    assert quaternion.shape[-1] == 4
    if normalize:
        quaternion = tf.math.l2_normalize(quaternion, axis=-1)
    if assert_normalized:
        quaternion = assert_normalized_quaternion(quaternion)

    # aliases
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    # rotation matrix from quaternion
    r11 = 1 - 2 * (y * y + z * z)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)

    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x * x + z * z)
    r23 = 2 * (y * z - x * w)

    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x * x + y * y)

    tf_transform_0 = tf.stack([r11, r21, r31], axis=-1)
    tf_transform_1 = tf.stack([r12, r22, r32], axis=-1)
    tf_transform_2 = tf.stack([r13, r23, r33], axis=-1)
    return tf.stack([tf_transform_0, tf_transform_1, tf_transform_2], axis=-1)


def rotation_matrix_to_quaternion(
    rotation_matrix: tf.Tensor, assert_valid: bool = False
):
    r = rotation_matrix
    assert r.shape[-2:] == [3, 3]
    if assert_valid:
        r = assert_valid_rotation(r)

    # aliases
    r11, r12, r13 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    r21, r22, r23 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    r31, r32, r33 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]

    nu1 = r11 + r22 + r33
    q1_a = 0.5 * tf.sqrt(1.0 + nu1)
    q1_b = 0.5 * tf.sqrt(
        (tf.square(r32 - r23) + tf.square(r13 - r31) + tf.square(r21 - r12))
        / (3.0 - nu1)
    )
    q1 = tf.where(nu1 > 0.0, q1_a, q1_b)

    nu2 = r11 - r22 - r33
    q2_a = 0.5 * tf.sqrt(1.0 + nu2)
    q2_b = 0.5 * tf.sqrt(
        (tf.square(r32 - r23) + tf.square(r12 + r21) + tf.square(r31 + r13))
        / (3.0 - nu2)
    )
    q2 = tf.where(nu2 > 0.0, q2_a, q2_b)

    nu3 = -r11 + r22 - r33
    q3_a = 0.5 * tf.sqrt(1.0 + nu3)
    q3_b = 0.5 * tf.sqrt(
        (tf.square(r13 - r31) + tf.square(r12 + r21) + tf.square(r23 + r32))
        / (3.0 - nu3)
    )
    q3 = tf.where(nu3 > 0.0, q3_a, q3_b)

    nu4 = -r11 - r22 + r33
    q4_a = 0.5 * tf.sqrt(1.0 + nu4)
    q4_b = 0.5 * tf.sqrt(
        (tf.square(r21 - r12) + tf.square(r31 + r13) + tf.square(r32 + r23))
        / (3.0 - nu4)
    )
    q4 = tf.where(nu4 > 0.0, q4_a, q4_b)

    pos = tf.ones_like(q1)
    neg = -tf.ones_like(q1)
    # assume q1 is positive
    q2_sign = tf.where((r32 - r23) > 0.0, pos, neg)
    q3_sign = tf.where((r13 - r31) > 0.0, pos, neg)
    q4_sign = tf.where((r21 - r12) > 0.0, pos, neg)
    q2 *= q2_sign
    q3 *= q3_sign
    q4 *= q4_sign
    return tf.stack([q1, q2, q3, q4], axis=-1)


def conjugate_quaternion(quaternion: tf.Tensor):
    """

    :param quaternion:
    :return:
    """
    return tf.concat(
        (quaternion[..., 0:1], tf.math.negative(quaternion[..., 1:4])), axis=-1
    )


def invert_quaternion(quaternion: tf.Tensor):
    """

    :param quaternion:
    :return:
    """
    return tf.math.l2_normalize(conjugate_quaternion(quaternion), axis=-1)


def multiply_quaternions(q1: tf.Tensor, q2: tf.Tensor):
    w1, w2 = q1[..., 0], q2[..., 0]
    v1, v2 = q1[..., 1:4], q2[..., 1:4]

    v = w1[:, None] * v2 + w2[:, None] * v1 + tf.linalg.cross(v1, v2)
    w = w1 * w2 - tf.reduce_sum(v1 * v2, axis=-1)
    return tf.concat((w[:, None], v), axis=-1)


def quaternion_rotation_angle(quaternion: tf.Tensor):
    return 2.0 * tf.math.atan2(
        tf.linalg.norm(quaternion[..., 1:4], axis=-1), quaternion[..., 0]
    )
