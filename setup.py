from setuptools import find_packages,setup
from typing import List

''' 
packages and distribute python project
contains metadata, dependencies 
'''

requirements_list:List[str]=[]
#find_packages finds __init__.py in each folder and packages them
def get_requirements()->List[str]:
    '''
    This function will return list of requirements.txt
    '''
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!='-e .':
                    requirements_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirements_list

# print(get_requirements())

setup(
    name="TelecomChurn",
    version="0.0.1",
    author="Samiulla",
    author_email="mdsamiulla2002@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
