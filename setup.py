from setuptools import find_packages, setup

def get_requirements(path):
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(path, 'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[x.strip() for x in requirements if x.strip() != "-e ."]
    
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Aishwarya Karwal',
    author_email='aishwaryakarwal1009@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
)