from setuptools import setup

setup(name='tf_quat2rot',
      version='1.0.3',
      description='Convert between quaternion and rotation matrices in tensorflow.',
      url='http://github.com/risteon/tf_quat2rot',
      author='Christoph Rist',
      author_email='c.rist@posteo.de',
      license='MIT',
      packages=['tf_quat2rot'],
      zip_safe=False,
      install_requires=[],
      test_suite='nose.collector',
      tests_require=['nose'],
      python_requires='>=3.5',
      extras_require={
          'tf': ['tensorflow'],
          'tf_gpu': ['tensorflow-gpu'],
      },
      )
