from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import os
from model import object_detection 

app = Flask(__name__)
app.secret_key = 'phuong0702'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/image_html/<path:filename>')
def send_image(filename):
    return send_from_directory('image_html', filename)

@app.route('/')
def home():
    session['username'] = 'phuong0702'
    return render_template('detected_image.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        object_detection("static/uploads/" + filename)
        return redirect(url_for('detection', filename=filename))
    
    return redirect(request.url)

@app.route('/results/<filename>')
def detection(filename):
    file_url = url_for('static', filename=f'results/{filename}')
    return render_template('detected_image.html', file_url=file_url)

if __name__ == '__main__':
    app.run(debug=True)
