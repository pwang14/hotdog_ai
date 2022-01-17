from flask import Flask, render_template, request, redirect
import os
from PIL import Image

import model

UPLOAD_FOLDER = os.path.join('static', 'temp_files')
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        if filename != '' and filename.split('.')[-1] in ALLOWED_EXTENSIONS:
            files = os.listdir(app.config['UPLOAD_FOLDER'])
            if len(files) > 5:
                for existing_filename in files:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], existing_filename))

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            guess = model.apply(path)

            return render_template('index.html', guess=guess, img_path=path)
        else:
            return render_template('index.html', guess=-1, error='Error receiving image')
    else:
        return render_template('index.html', guess=-1)

if __name__ == "__main__":
    app.run(debug=True)