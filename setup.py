from setuptools import setup, find_packages

setup(
    name='organoid_analyzer',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/acorbat/organoid_analyzer/tree/master/',
    license='MIT',
    author='Agustin Corbat',
    author_email='acorbat@df.uba.ar',
    description='Segmenter and morphological analyzer for organoids.',
    install_requires=['imageio', 'IPython', 'matplotlib', 'numpy', 'pandas',
                      'mahotas', 'pyyaml', 'scikit-image', 'scikit-learn',
                      'scipy', 'seaborn',
                      'img_manager @ git+https://github.com/acorbat/img_manager.git',
                      'serialize @ git+https://github.com/hgrecco/serialize.git']
)