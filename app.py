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
    # Retrieve data from temp or just re-run the detection for the report.
    # For a production app we'd keep this in a session/db. 
    # Here we simply rerun it to generate the report data fresh.
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
        
    results = detect_microplastics(filepath, app.config['ANNOTATED_FOLDER'])
    results['original_image'] = filename
    
    report_html = render_template('downloadable_report.html', **results)
    
    mem = io.BytesIO()
    mem.write(report_html.encode('utf-8'))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"Microplastic_Analysis_Report_{filename}.html", mimetype='text/html')

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    with open("error_log.txt", "w") as f:
        f.write(traceback.format_exc())
    return f"Internal Server Error! See error_log.txt or ask AI. Details: {str(e)}", 500
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
