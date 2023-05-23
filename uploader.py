from flask import Flask, request, render_template
import os

app = Flask(__name__)

# set the upload folder and allowed file types
UPLOAD_FOLDER = r"D:\PDARS\WebApp\uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # get the checkbox value and create a directory with that name
        checkbox_value = request.form.get('checkbox')
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], checkbox_value)):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], checkbox_value))

        # get the uploaded file and save it to the directory
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], checkbox_value, filename))
            return 'File successfully uploaded, Response Registered!'
    # render the HTML template with the checkboxes
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
