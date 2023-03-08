from setuptools import setup, find_packages

min_requirements = [
    'numpy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'scikit-optimize',
    'lightgbm',
    'catboost',
    'seaborn',
    'scipy',
    'scikit-optimize',
    'tabulate',
    'scikit-plot'
    ]



setup(

    name='automate_ML',
    version="1.0.4",

    author='Sohaib Umer',
    author_email='sohaibfccu@gmail.com',

    description='A python module to solve machine learning problems in a mechanized way. The repository is able to preprocess the data and output the results in numerical as well as in the graphical form.',
    long_description='https://github.com/sohaibunist/automate_ML',
    long_description_content_type="text/markdown",


    packages=find_packages(),
    install_requires=min_requirements,

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3'

    ],

)
