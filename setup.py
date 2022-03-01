from setuptools import setup

setup(
    name='PyPBG',
    version='0.0.1',
    description='Propagation in Bipartite Graph',
    url='https://github.com/thiagodepaulo/PyPBG',
    author='Thiago de Paulo Faleiros',
    author_email='thiagodepaulo@gmail.com',
    license='',
    packages=['pbg'],
    install_requires=['sklearn',
                      'numpy',
                      'tqdm',
                      'nltk',
                      'pandas',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
