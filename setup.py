from setuptools import find_packages, setup
import setuptools.command.build_py
import subprocess
import os.path

VERSION = '0.1.0'
"""Package version"""

class BuildWithDVC(setuptools.command.build_py.build_py):
  """
  Downloads model artifacts stored in DVC but not in the source code if the artifacts are not present.

  This is needed when running pip install git+https as the training outputs are not present.
  """

  def run(self):
      self._dvc_get('src/disaster_tweets/artifacts/model.h5')
      self._dvc_get('src/disaster_tweets/artifacts/tokenizer.pickle')
      setuptools.command.build_py.build_py.run(self)

  def _dvc_get(self, path):
      if os.path.isfile(path):
        return
      # This should really grab the artifacts at the associated package version. However, if downloading direct from Git,
      # there isn't a clean way to specify this. Just grabbing from master for now.
      git_revision = 'v'+VERSION
      subprocess.check_call(['dvc get --rev {} -o {} https://github.com/whisk-ml/disaster_tweets {}'.format(git_revision,path,path)], shell=True)

setup(
    cmdclass={
        'build_py': BuildWithDVC,
    },
    name='disaster_tweets',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    version=VERSION,
    include_package_data=True,
    # By default only whisk is added as a dependency. whisk is required for
    # accessing whisk.project.artifacts_dir in the default models.model.Model class.
    # You likely need to list more dependencies for your model package.
    # For example: your model framework (Scikit, Torch, etc), numpy, Pandas, etc.
    #
    # Copying the requirements.txt dependencies into `install_requires=` isn't recommended as this
    # includes many dependencies that are not required to generate predictions and can result in a very large package.
    #
    # You can test that the package works and contains needed dependencies by running `tox` from the
    # command line. tox tests the package in an isoloated venv.
    install_requires=[
        'whisk==0.1.27', 'keras>=2.3.1', 'tensorflow-cpu', 'dvc'
    ],
    entry_points={
        'console_scripts': [
            'disaster_tweets=disaster_tweets.cli.main:cli',
        ],
    },
    description='A short description of the project.',
    author='Your name (or your organization/company/team)',
    author_email='you@example.com',
    python_requires='>=3.6',
    url="https://ADD THE URL TO YOUR PROJECT GITHUB REPO OR DOCS"
)
