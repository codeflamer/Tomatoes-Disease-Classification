import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = 'tomatoes-tensor-model'
CLASSNAME = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

model = None

def download_blob(bucket_name,source_blob_name,destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/tomatoes3.h5",
            "/tmp/tomatoes3.h5"
        )
        model = tf.keras.models.load_model("/tmp/tomatoes3.h5")
    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image / 255
    img_array = image[tf.newaxis,:]
    predictions = model.predict(img_array)

    predicted_class = CLASSNAME[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions)),2)

    return {"prediction_class":predicted_class,"confidence":confidence}
