from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Package for running WFSC algorithms on optical models or instruments'
LONG_DESCRIPTION = 'Package for running WFSC algorithms on optical models or instruments'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="centaur",
        version=VERSION,
        author="Kian Milani",
        author_email="<kianmilani@arizona.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'WFSC'],
        classifiers= [
            "Development Status :: Alpha-0.1.0",
            "Programming Language :: Python :: 3",
        ]
)
