import tensorflow as tf
from tensorflow import keras 
import cv2 
import glob 
import numpy as np

keras.models.load_model("model.h5")

# control = [cv2.imread(image) for image in glob.glob("./training_data/control/*.jpg")]
# kills = [cv2.imread(image) for image in glob.glob("./training_data/kills/*.jpg")]

control = cv2.imread("./control.png")
control = cv2.resize(control, (227, 227))
img = tf.keras.utils.img_to_array(control)
img = tf.expand_dims(img, 0)

model = tf.keras.models.load_model('model.h5')
probs = model.predict((img))

class_names = ["Control", "Kills"]

score = tf.nn.softmax(probs[0])

print(score)
print(class_names[np.argmax(score)])

