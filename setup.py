from setuptools import setup, find_packages

setup(
    name='proportion_localised',  
    version='0.1.3',
    description="Proportion Localised is a novel metric for visual anomaly detection, intended to be more interpretable than existing metrics. Published in Transactions of Machine Learning Research (TMLR) 2024: VisionAD, a software package of performant anomaly detection algorithms, and Proportion Localised, an interpretable metric.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Alex Taylor',
    author_email='alex.taylor07@hotmail.co.uk',
    url='https://github.com/alext1995/proportion-localised',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)