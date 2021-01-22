from setuptools import setup, find_packages

pkgs = ['Keras==2.2.4',
	'nibabel==2.5.0',
	'numpy==1.17.0',
	'pydicom==1.3.0',
	'pynrrd==0.4.0',
	'scikit-learn==0.21.3',
	'scikit-image==0.15.0',
	'requests==2.22.0',
	'opencv-python==4.1.0.25',
	'pandas==0.25.0',
	'xlrd==1.1.0',
	'medpy==0.4.0',
        'h5py==2.10.0']

pkgs.append('tensorflow==1.13.1')
# pkgs.append('tensorflow-gpu==1.15')

setup(name='lung_segmentation',
      version='1.0',
      description='Application to segment lungs using Deep Learning',
      url='https://github.com/TransRadOnc-HIT/lung_segmentation.git',
      python_requires='>=3.5',
      author='Francesco Sforazzini',
      author_email='f.sforazzini@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=pkgs,
      entry_points={
          'console_scripts': ['run_lung_segmentation = scripts.run_inference:main',
			      'run_segmentation_training = scripts.run_training:main']},
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
