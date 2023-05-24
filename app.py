import os
import uuid
import flask
import urllib
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D ,Dropout ,BatchNormalization
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau , ModelCheckpoint
from keras.applications.mobilenet import MobileNet ,preprocess_input
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
#tensorflow.data.experimental.enable_debug_mode()
#tensorflow.config.run_functions_eagerly(True)
optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.99)
from flask import Flask , render_template  , request , send_file, Response
from keras_preprocessing.image import load_img , img_to_array
import cv2
import base64
import pandas as pd
import numpy as np
import sys
sys.path.append("./lib")
import Dataloader
import modeler
import pathlib
import random
import gunicorn

gunicorn_options = {
    'bind': '127.0.0.1:8000',  # Replace with your desired host and port
    'workers': 4,  # Number of worker processes
    'timeout': 180,  # Worker timeout in seconds
}

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR , 'mobilenet_final.hdf5'))
camera = cv2.VideoCapture(0)
# Read in the CSV file
df = pd.read_csv('Pesticides.csv')


i=0
# set the upload folder and allowed file types
UPLOAD_FOLDER = './uploads'
#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['Adristyrannus','Aleurocanthus spiniferus','alfalfa plant bug','alfalfa seed chalcid','alfalfa weevil','Ampelophaga','aphids',
           'Aphis citricola Vander Goot','Apolygus lucorum','army worm','asiatic rice borer','Bactrocera tsuneonis','beet army worm','beet fly',
           'Beet spot flies','beet weevil','beetle','bird cherry-oataphid','black cutworm','Black hairy','blister beetle','bollworm',
           'Brevipoalpus lewisi McGregor','brown plant hopper','cabbage army worm','cerodonta denticornis','Ceroplastes rubens','Chlumetia transversa',
           'Chrysomphalus aonidum','Cicadella viridis','Cicadellidae','Colomerus vitis','corn borer','corn earworm','cutworm','Dacus dorsalis(Hendel)',
           'Dasineura sp','Deporaus marginatus Pascoe','english grain aphid','Erythroneura apicalis','fall armyworm','Field Cricket','flax budworm',
           'flea beetle','Fruit piercing moth','Gall fly','grain spreader thrips','grasshopper','green bug','grub','Icerya purchasi Maskell','Indigo caterpillar',
           'Jute aphid','Jute hairy','Jute red mite','Jute semilooper','Jute stem girdler','Jute Stem Weevil','Jute stick insect','large cutworm',
           'Lawana imitata Melichar','Leaf beetle','legume blister beetle','Limacodidae','Locust','Locustoidea','longlegged spider mite','Lycorma delicatula',
           'lytta polita','Mango flat beak leafhopper','meadow moth','Mealybug','Miridae','mites','mole cricket','Nipaecoccus vastalor','odontothrips loti',
           'oides decempunctata','paddy stem maggot','Panonchus citri McGregor','Papilio xuthus','parathrene regalis','Parlatoria zizyphus Lucus','peach borer',
           'penthaleus major','Phyllocnistis citrella Stainton','Phyllocoptes oleiverus ashmead','Pieris canidia','Pod borer','Polyphagotars onemus latus',
           'Potosiabre vitarsis','Prodenia litura','Pseudococcus comstocki Kuwana','red spider','Rhytidodera bowrinii white','rice gall midge','rice leaf caterpillar',
           'rice leaf roller','rice leafhopper','rice shell pest','Rice Stemfly','rice water weevil','Salurnis marginella Guerr','sawfly','Scirtothrips dorsalis Hood',
           'sericaorient alismots chulsky','small brown plant hopper','Spilosoma Obliqua','stem borer','Sternochetus frigidus','tarnished plant bug','Termite',
           'Termite odontotermes (Rambur)','Tetradacus c Bactrocera minax','therioaphis maculata Buckton','Thrips','Toxoptera aurantii','Toxoptera citricidus',
           'Trialeurodes vaporariorum','Unaspis yanonensis','Viteus vitifoliae','wheat blossom midge','wheat phloeothrips','wheat sawfly','white backed plant hopper',
           'white margined moth','whitefly','wireworm','Xylotrechus','yellow cutworm','Yellow Mite','yellow rice borer']


def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    img = img_to_array(img)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)
    #result = np.argmax(result)
    dict_result = {}
    for i in range(132):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    #finding Top three probabailities
    prob_result = []
    class_result = []
    #image_path = r"D:\PDARS\version2.1.9\new_dataset - Copy\DATASET\white margined moth\11817.jpg"
    #folder_path = r"D:\PDARS\version2.1.9\new_dataset - Copy\DATASET"
    hash = hash_sum(filename, DATASET)
    print(result)
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result, hash
DATASET = './DATASET'
@app.route('/retrainer' , methods = ['GET' , 'POST'])
def retrainer(path = DATASET):
    training, testing, validing = Dataloader.dataloader(path)
    model = load_model(os.path.join(BASE_DIR , 'mobilenet_final.hdf5'))
    mobilenet=MobileNet(include_top=False,weights='imagenet',input_shape=(224,224,3))
    mobilenet.trainable=False
    mob_model=Sequential([
        mobilenet,
        MaxPooling2D(3,2),
        Flatten(),
        Dense(128,activation='relu'),
        BatchNormalization(),
        Dense(1024,activation='relu'),
        BatchNormalization(),
        Dense(512,activation='relu'),
        BatchNormalization(),
        Dense(132,activation='softmax')
    ])
    mob_model.compile(optimizer=optimizer,loss='categorical_crossentropy', run_eagerly=False, metrics=["accuracy", "Precision", "Recall", "AUC"])
    epochs = 10
    batch_size=32
    steps_per_epoch = training.n // batch_size
    validation_steps = validing.n // batch_size
    
    print("[INFO] Starting Model Training")
    history_mob=mob_model.fit(training,validation_data=validing,epochs=1,batch_size=batch_size,
                          steps_per_epoch=steps_per_epoch,validation_steps=validation_steps, verbose=1)
    
    print("[INFO] Model training Complete")
    #return history_mob
    
    print("[INFO] Saving Model")
    try:
        pathlib.Path ("mobilenet_retrained.hdf5").unlink ()
        mob_model.save('mobilenet_retrained.hdf5')
        model = load_model(os.path.join(BASE_DIR , 'mobilenet_retrained.hdf5'))
    except:
        model = load_model(os.path.join(BASE_DIR , 'mobilenet_retrained.hdf5'))
    print("[INFO] Retrained Model Loaded")
    
    return "Model Training complete, Please return to homepage"
    
    
@app.route('/stoptrainer' , methods = ['GET' , 'POST'])
def stoptrainer():
    return render_template('index.html')

def perpetual_hash(imageA, imageB):
    # err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err = np.sum((imageA.astype("float") - imageB[:, :, :3].astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def hash_sum(img, path):
    input_image = cv2.imread(img)
    for root, dirs, files in os.walk(path):
        for file in files:
            image = cv2.imread(os.path.join(root, file))
            image_resized = cv2.resize(image, input_image.shape[:2][::-1])
            if perpetual_hash(input_image, image_resized) < 50:
                hash = os.path.basename(root)
                return hash
    return None



@app.route('/')
def home():
        return render_template("index.html")

def checksum(class_result, prob_result, hash):
    class_result[0] = hash
    predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "prob1": (100-prob_result[0]),
                    "prob2": (100-prob_result[1]),
                    "prob3": (100-prob_result[2]),
                }
    return predictions
    
@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result, hash = predict(img_path , model)
                if ((hash != class_result[0]) and (hash != None)):
                    checksum(class_result , prob_result, hash)
                    return predictions

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result, hash = predict(img_path , model)
                if (hash != class_result[0] and (hash != None)):
                    checksum(class_result , prob_result, hash)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }
                # Select the index of the row where the first column equals the string "my_string"
                row_index = (df.iloc[:, 0].str.lower() == class_result[0].lower()).idxmax()

                # Print the output message
                pesticide_text = f"Common pesticides used for controlling {df.iloc[row_index, 0]} are {df.iloc[row_index, 1]}"
                
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions, pesticide_text=pesticide_text)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

@app.route('/camera', methods=['GET', 'POST'])
def cam():
    return render_template('cam.html')

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

# set the upload folder and allowed file types
UPLOAD_FOLDER = './DATASET'
REPOSITORY  = './uploads'
#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/feedbacker', methods=['GET','POST'])
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
            #retrainer(path = DATASET)
            return render_template('loader.html')
            
    # render the HTML template with the checkboxes
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug = False)


