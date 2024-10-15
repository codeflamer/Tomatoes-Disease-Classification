from fastapi import FastAPI,UploadFile
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MODEL = tf.keras.models.load_model("./tomatoes3.h5")

origins = [
    "http://localhost:3000",
    "https://ml-projects-frontend.vercel.app/"

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def default_page():
    return {"status_code": 200, "message": "OK"}

@app.post("/predict")
async def predict(file:UploadFile):
    image = await file.read()
    image = BytesIO(image)
    image = np.array(Image.open(image))
    prediction = MODEL.predict(image[tf.newaxis,:])
    predict_class = CLASSNAME[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {"prediction_class":predict_class,"confidence" : confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)