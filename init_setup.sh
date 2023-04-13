echo [$(date)]: "START" 
echo [$(date)]: "creating env with python 3.10 version" 
conda create --prefix ./env python=3.10 -y
echo [$(date)]: "activating the environment" 
conda activate ./env
echo [$(date)]: "installing the dev requirements" 
pip install -r requirements_dev.txt
echo [$(date)]: "END" 
echo $CONDA_DEFAULT_ENV