# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = 'c.rist@posteo.de'

import tf_quat2rot
import tensorflow as tf


class TestGraphMode(tf.test.TestCase):

    @tf.function
    def _run_in_graph(self, batch_shape=(2, 1, 3)):
        random_quats = tf_quat2rot.random_uniform_quaternion(batch_dim=batch_shape)
        random_rotations = tf_quat2rot.quaternion_to_rotation_matrix(random_quats)
        random_quats_restored = tf_quat2rot.rotation_matrix_to_quaternion(random_rotations)
        return random_quats, random_quats_restored

    def test_graph_mode(self):
        with self.session(use_gpu=False):
            # single random quaternion
            batch_shape = (2, 1, 3)
            random_quats, random_quats_restored = self._run_in_graph(batch_shape)

            self.assertEqual(batch_shape + (4,), random_quats_restored.shape)
            self.assertAllClose(random_quats, random_quats_restored)
