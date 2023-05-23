from flask import Flask, render_template, Response, request
import cv2
import os
import base64

app = Flask(__name__)

camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        cv2.imwrite('captured_image.jpg', frame)
        return 'Image captured successfully!'
    else:
        return 'Failed to capture image.'

@app.route('/save', methods=['POST'])
def save():
    # Get the base64-encoded image data from the form
    data_url = request.form['imageData']
    # Remove the prefix and save the remaining data
    img_data = base64.b64decode(data_url.split(',')[1])
    # Save the image data to a file
    with open('capture.jpg', 'wb') as f:
        f.write(img_data)
    # Show a message indicating that the image was saved
    return 'Image saved as capture.jpg'


if __name__ == '__main__':
    app.run(debug=True)
