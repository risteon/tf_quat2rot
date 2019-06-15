# -*- coding: utf-8 -*-

__author__ = """Christoph Rist"""
__email__ = 'c.rist@posteo.de'

from .conversion import quaternion_to_rotation_matrix
from .conversion import rotation_matrix_to_quaternion


__all__ = [
    'quaternion_to_rotation_matrix',
    'rotation_matrix_to_quaternion',
]
