"""
author @ kumar dahal
this file is created on folders structure and files.
"""

import os
from pathlib import Path
import logging


#level of logging is INFORMATION and format we are using is asctime and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')

package_name = "movie_predictor"

list_of_files = [
   ".github/workflows/.gitkeep",  #for empty files we add .gitkeep because empty folder will  never be created on github
   #every folder will be inside source repoistory which is our main folder
   f"src/{package_name}/__init__.py", 
   f"src/{package_name}/components/__init__.py", #for writing each component
   f"src/{package_name}/utils/__init__.py", #this is our utility folder
   f"src/{package_name}/config/__init__.py", #config contains configuration
   f"src/{package_name}/pipeline/__init__.py", 
   f"src/{package_name}/entity/__init__.py", 
   f"src/{package_name}/constants/__init__.py",
   #these are our test file 
   "tests/__init__.py",
   "tests/unit/__init__.py",
   "tests/integration/__init__.py",
   
   "config/config.yaml",
   "dvc.yaml",#data version control
   "params.yaml",#contains our paramter for training
   "init_setup.sh",#shell script for creating environment,since we oftain activate environment and install this is for onetime run
   "requirements.txt", 
   "requirements_dev.txt",#requirements for development
   "setup.py",#for creating python package 
   "setup.cfg",#for creating python package 
   "pyproject.toml",#for creating python package 
   "tox.ini",#for testing of project locally
   "research/trials.ipynb", #for trails of project ipynb files
]
#creating this files
for filepath in list_of_files:

    #pass filepath through Path library which will create OS indipindent file
    
    filepath = Path(filepath)


    #since some  of the file have folder structure
    #os.path. split split the filepath with filename and filepath

    filedir, filename = os.path.split(filepath)

    #if filedir is not present then create those directory 

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        #logging to see log 
        logging.info(f"Creating file directory: {filedir} for file: {filename}")
    
    #check if file exist or not
    #if size of file is zero and not exist then create emppty 
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} already exists")