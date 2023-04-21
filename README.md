#MoviePredict
<h1>This is the class Project of AISC1006 </h1> <br/>

<h5> Create environment using conda from terminal in pycharm</h5>
1) conda create -p envName python==3.9 -y  <br/>
check conda is working or not using conda --version <br/>
add environment veriable if not working then <br/>

activate conda using conda activate envname <br/>

<h3>create requirements.txt </h3> <br/>
<h3> create app.py </h3> <br/>
<h3> run app.py </h3> <br/>

<h3>BUILD DOCKER IMAGE</h3>

docker build -t image_name:tagName . <br/>

to list docker images <br/>
docker images <br/>
to run image <br/>
docker run -p 5000:5000 -e PORT=5000 imageID <br/>


<h3>to check running container </h3> <br/>
docker ps <br/>
<h3> to stop container  </h3><br/>
docker stop container_id <br/>

 <h3> run  setup.py </h3> to install the requirements <br/>
<h3>python setup.py install</h3> <p style="color:red;"> no need to install requirements.txt file from  now</p></br>
The aim of the setup is to give description of the project and install the requirements along with the packages in mvoie_predictor <br/>
It creates the .egg folder and have the source information and package information of the projects <br/>

 

<h3>-e .</h3> is used while we have setup.py  file for installing package under the current directory </br>
<h3>remove("-e .")</h3> is used because in setup.py we already have find_package()</br>



<h3> Create  the folders inside mvoie_predictor Package</h3>
logger: for writing the logs of what is going on </br>

<h3>Created the Entity of the project</h3>
"/home/$USER/Documents/mvoie_predictor/mvoie_predictor/entity/config_entity.py" </br>
entity need the properities ,so these properities are defined on </br>
"/home/$USER/Documents/mvoie_predictor/config/config.yaml"  </br> 

<h4> In order to read this yaml file we  need yaml file reader which is defined on </h4></br>
 "/home/$USER/Documents/mvoie_predictor/mvoie_predictor/utils/yaml_reader.py"

<h4>we have entity, properities ,and reader Now, we can read yaml file and assign it the properities ,which is done on </h4></br>
"/home/$USER/Documents/mvoie_predictor/mvoie_predictor/config/configurations.py" 

<h2> Evidently AI library is used for checking data drift </h2>
