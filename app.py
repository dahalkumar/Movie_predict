"""
author @ kumar dahal
this code is written to run flask server and pipeline
"""
from flask import Flask, request
import sys

import pip
from movie_prediction.utils.common import read_yaml_file, write_yaml_file
from matplotlib.style import context
from movie_prediction import logging
from movie_prediction.exception import MovieException
import os, sys
import json
from movie_prediction.config.configuration import Configuartion
from movie_prediction.constants import CONFIG_DIR, get_current_time_stamp
from movie_prediction.pipeline.pipeline import Pipeline
from movie_prediction.entity.movie_predictor import MoviesPredictor, MovieData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "src"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from movie_prediction import get_log_dataframe

MOVIES_DATA_KEY = "movie_data"
WORLD_REVENUE_KEY = "world_revenue_value"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'movies'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("movies", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    mpaa_options = ['PG-13', 'R', 'Not Rated', 'G', 'PG']
    if request.method == 'POST':
        data = {
            'originalTitle': request.form['originalTitle'],
            'distributor': request.form['distributor'],
            'opening_theaters': request.form['opening_theaters'],
            'budget': request.form['budget'],
            'MPAA': request.form['MPAA'],
            'release_days': request.form['release_days'],
            'startYear': request.form['startYear'],
            'runtimeMinutes': request.form['runtimeMinutes'],
            'genres_y': request.form['genres_y'],
            'averageRating': request.form['averageRating'],
            'numVotes': request.form['numVotes'],
            'ordering': request.form['ordering'],
            'category': request.form['category'],
            'primaryName': request.form['primaryName'],
          
        }
        # do something with the data, such as make a prediction
        return f'Prediction for {data["originalTitle"]} with MPAA {data["MPAA"]}'
    else:
        return render_template('eda.html', mpaa_options=mpaa_options)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        MOVIES_DATA_KEY: None,
        WORLD_REVENUE_KEY: None
    }


    
    if request.method == 'POST':
        originalTitle = request.form['originalTitle']
        opening_theaters = float(request.form['opening_theaters'])
        budget = float(request.form['budget'])
        MPAA = request.form['MPAA']
        release_days = float(request.form['release_days'])
        startYear = float(request.form['startYear'])
        runtimeMinutes = float(request.form['runtimeMinutes'])
        genres_y = request.form['genres_y']
        averageRating = float(request.form['averageRating'])
        numVotes = float(request.form['numVotes'])
        ordering = float(request.form['ordering'])
        category = request.form['category']
        primaryName = request.form['primaryName']
        distributor = request.form['distributor']

        movies_data = MovieData(originalTitle=originalTitle,
                                   opening_theaters=opening_theaters,
                                   budget=budget,
                                   MPAA=MPAA,
                                   release_days=release_days,
                                   startYear=startYear,
                                   runtimeMinutes=runtimeMinutes,
                                   genres_y=genres_y,
                                   averageRating = averageRating,
                                   numVotes=numVotes,
                                   ordering=ordering,
                                   category=category,
                                   primaryName = primaryName,
                                   distributor=distributor
                                   )
        movies_df = movies_data.get_movies_input_data_frame()
        movies_predictor = MoviesPredictor(model_dir=MODEL_DIR)
        world_revenue_value = movies_predictor.predict(X=movies_df)
        context = {
            MOVIES_DATA_KEY: movies_data.get_movies_data_as_dict(),
            WORLD_REVENUE_KEY: world_revenue_value,
        }
        for key, values in context[MOVIES_DATA_KEY].items():
            values_str = ', '.join(str(v) for v in values)
            print(f"{key}: {values_str}")
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8001)
