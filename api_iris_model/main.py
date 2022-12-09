from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os
app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:5000",
# ]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##################### Load Model #########################
def load_model():
    os.system('ls')
    # return joblib.load("api_iris_model/model/iris_classifier.joblib") #Path without Docker
    return joblib.load("model/iris_classifier.joblib") #To Dockerfile workdir

##################### BaseModel ###########################
class InferenceParameters (BaseModel):
    """Medidas de las carecteristicas de la flor seaparadas por coma

    Args:
        medidas (string): String de medidas
    
    Ejemplo
    =======
    6.3, 2.7, 4.9, 1.8
    """
    medidas: list

####################### EndPoints ##########################

@app.post("/predict/")
async def inference(input: InferenceParameters):
    clf = load_model()
    df_input = pd.DataFrame([input.medidas])
    result = clf.predict(df_input)
    
    return{"input": input, "result": float(result[0]) }

@app.get("/")
async def root():
    model = load_model()
    return {"message": "OK"}