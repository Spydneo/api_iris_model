from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import pickle
import pandas as pd
import os


app = FastAPI()

##################### ENUM #########################
class TitanicSex(str, Enum):
    """Valores posibles del titanic """
    MALE = "male"
    FEMALE = "female"

class OutputValues(str, Enum):
    """Tipo de salida del modelo """
    PROBABILITY = "prediction"
    PREDICTION = "probability"


##################### Cargar pickle #########################
def load_model():
    # os.system('ls')
    return joblib.load("api_iris_model/model/iris_classifier.joblib")

# with open(f"./titanic_model_test.pkl", 'rb') as f:
#     pipeline_cargado_1 = pickle.load(f)

##################### BaseModel ###########################
class ModelConfig (BaseModel):
    """Configuracíon del modelo"""
    output: OutputValues
class TitanicModelInput(BaseModel):
    """Modelo para reibir la X de un model de prección del Titanic """
    sex: TitanicSex
    age: int
class InferenceParameters (BaseModel):
    """Input de la API"""
    titanic_model_input: TitanicModelInput
    model_config: ModelConfig

####################### EndPoints ##########################

@app.post("/predict/")
# async def create_item(input: InferenceParameters, user_id: int):
async def inference(input: list):
    clf = load_model()

    df_input = pd.DataFrame(
                            [input]
                            )

    result = clf.predict(df_input)
    print(result)
    return{"whole_input": input, "result": float(result[0]) }



@app.get("/")
async def root():
    model = load_model()
    return {"message": "OK"}

@app.get("/echo/{number_id}")
async def read_number(number_id: int) -> str:
    return {"n": str(number_id)}
