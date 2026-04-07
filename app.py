import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ANNOTATED_FOLDER'] = 'static/annotated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)

from processing.detect import detect_microplastics
import json
from flask import send_file
import io


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = detect_microplastics(filepath, app.config['ANNOTATED_FOLDER'])
        
        # Add original filename to results for display
        results['original_image'] = filename
        
        # Store results in session or temporarily for download
        # A simple hack for download without session is passing it
        
        return render_template('result.html', **results)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download/<filename>')
def download(filename):
    # This route just generates a dummy PDF or text report on the fly
    # We can use io to send a simple text file
    report_content = f"Microplastic Analysis Report for {filename}\nContamination check completed.\nPlease view main results on the dashboard."
    mem = io.BytesIO()
    mem.write(report_content.encode('utf-8'))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"report_{filename}.txt", mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
