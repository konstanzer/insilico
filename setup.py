from setuptools import setup

long_description = open("README.md").read()

setup(
    name='insilico',
    version='0.1.2',
    description='A Python package to process & model ChEMBL data.',
    long_description_content_type="text/markdown",
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python'
        ],
    url='https://github.com/konstanzer/insilico',
    author='Steven Newton',
    author_email='steven.j.newton99@gmail.com',
    license='MIT',
    packages=['insilico'],
    install_requires=[],
    include_package_data=True,
    package_data={
        'insilico': ['fingerprints_xml/*', 'data/empty.txt'],
    },
    zip_safe=False
)