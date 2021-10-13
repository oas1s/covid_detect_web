from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import neuralnetwork
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/CT/"
IMAGE_FOLDER = os.path.join('static', 'images')


@app.route('/')
def upload_file_page():
    return render_template('uploadPage.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(IMAGE_FOLDER,secure_filename(f.filename))
            f.save(path)
            print(f.filename)
            prediction = neuralnetwork.predict(path)
            print(prediction)
            if prediction[0] == 1:
                return render_template('noncovid.html', filepath = path,prob = prediction[1])
            else:
                return render_template('covid.html', filepath = path,prob = prediction[1])

        else:
            return render_template("errorPage.html"), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
