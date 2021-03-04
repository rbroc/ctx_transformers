from setuptools import setup, find_packages
PACKAGES = find_packages()

__version__ = '0.0.1'


if __name__ == '__main__':

    setup(
        name="reddit",
        version=__version__,
        description="Personality encoders",
        maintainer='Roberta Rocca',
        maintainer_email='rbrrcc@gmail.com',
        url='http://github.com/rbroc/personality_reddit',
        install_requires=['numpy', 'pandas', 'tensorflow', 'keras', 'seaborn', 'official', 'transformers'],
        packages=find_packages(exclude=['tests']),
        license='MIT',
        zip_safe=False,
        download_url=(f"https://github.com/rbroc/personality_reddit/archive/{__version__}.tar.gz")
    )
