# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = 'c.rist@posteo.de'

from .converter import quaternion_to_rotation_matrix
from .converter import rotation_matrix_to_quaternion
from .generator import random_uniform_quaternion
from .generator import random_uniform_rotation_matrix


__all__ = [
    'quaternion_to_rotation_matrix',
    'rotation_matrix_to_quaternion',
    'random_uniform_quaternion',
    'random_uniform_rotation_matrix',
]
