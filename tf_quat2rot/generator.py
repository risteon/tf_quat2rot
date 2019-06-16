# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = 'c.rist@posteo.de'

import math
import tensorflow as tf
from .check import assert_normalized_quaternion, assert_valid_rotation
from .converter import quaternion_to_rotation_matrix


def random_uniform_quaternion(batch_dim=(), dtype=tf.float64, assert_normalized: bool = False):
    """ Generate rotations that are uniformly distributed over SO(3)

    Reference: http://planning.cs.uiuc.edu/node198.html
    """
    if isinstance(batch_dim, tf.Tensor):
        assert batch_dim.dtype == tf.int32
    else:
        batch_dim = tf.convert_to_tensor(batch_dim, dtype=tf.int32)
    assert batch_dim.ndim == 1

    u = tf.random_uniform(shape=tf.concat((batch_dim, (3, )), axis=0), dtype=dtype)
    sqrt_1_u0 = tf.sqrt(1 - u[..., 0])
    sqrt_u0 = tf.sqrt(u[..., 0])
    pi2_u1 = u[..., 1] * 2 * math.pi
    pi2_u2 = u[..., 2] * 2 * math.pi
    w = sqrt_1_u0 * tf.sin(pi2_u1)
    x = sqrt_1_u0 * tf.cos(pi2_u1)
    y = sqrt_u0 * tf.sin(pi2_u2)
    z = sqrt_u0 * tf.cos(pi2_u2)
    quats = tf.stack([w, x, y, z], axis=-1)

    # w should be positive. So flip sign if negative
    w_tiled = tf.tile(tf.expand_dims(w, axis=-1),
                      multiples=tf.concat((tf.ones(shape=tf.size(tf.shape(w)),
                                                   dtype=tf.int32), (4, )), axis=0))
    quats = tf.where(w_tiled < 0.0, tf.math.negative(quats), quats)
    if assert_normalized:
        quats = assert_normalized_quaternion(quats)
    return quats


def random_uniform_rotation_matrix(batch_dim=(), dtype=tf.float64, assert_valid: bool = False):
    q = random_uniform_quaternion(batch_dim, dtype, assert_normalized=False)
    r = quaternion_to_rotation_matrix(q)
    if assert_valid:
        r = assert_valid_rotation(r)
    return r
