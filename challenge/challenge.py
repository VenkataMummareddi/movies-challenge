import time

from flask import Flask,render_template,request
import os
import model

app = Flask(__name__)

app.config['DATA_SETS_FOLDER'] = 'static/datasets'

@app.route('/')
def open_app():
    return render_template('index.html')

@app.route('/genres/train',methods=["POST"])
def train_model():
    dataset_to_train = request.files['train_file']
    file_name = "train_" + str(time.time()) + ".csv"
    if dataset_to_train.filename != '':
        dataset_to_train.save(os.path.join(app.config['DATA_SETS_FOLDER'], file_name))
    model.train_model(file_name)
    return render_template('index.html', trained=True)

@app.route('/genres/predict',methods=["POST"])
def predict_genre():
    dataset_to_predict = request.files['predict_file']
    file_name = "predict_"+str(time.time())+".csv"
    if dataset_to_predict.filename != '':
        dataset_to_predict.save(os.path.join(app.config['DATA_SETS_FOLDER'], file_name))
    print(model.predict(file_name))
    return render_template('index.html', predicted=True,predictionResult="Comedy")

if __name__ == "__main__":
     app.run(port = 5000)