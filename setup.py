from setuptools import setup

setup(
    name='analysis',
    version='0.1',
    py_modules=['analysis'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        analysis=analysis:cli
    ''',
)
