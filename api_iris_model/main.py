from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pickle
import pandas as pd

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
with open(f"./titanic_model_test.pkl", 'rb') as f:
    pipeline_cargado_1 = pickle.load(f)

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
async def create_item(input: InferenceParameters):
    input.titanic_model_input.age
    df_input = pd.DataFrame(
                            [{
                            "age":input.titanic_model_input.age,
                            "sex":input.titanic_model_input.sex.value}]
                            )

    result = 1
    if(input.model_config.output.value == "prediction"):
        result = pipeline_cargado_1.predict(df_input)   
    elif(input.model_config.output.value == "probability"):
        result = pipeline_cargado_1.predict_proba(df_input)

    return{"whole_input": input, "result": list(result[0]) }
    # Primer ejercicio
    # if(user_id % 2 == 0):
    #     result = pipeline_cargado_1.predict(df_input)
    #     model = "Decission Tree"
    # else:
    #     result = pipeline_cargado_2.predict(df_input)
    #     model = "Lineal"

    # return{"whole_input": input, "result": result[0], "model": model}


@app.get("/")
async def root():
    return {"message": "OK"}

@app.get("/echo/{number_id}")
async def read_number(number_id: int) -> str:
    return {"n": str(number_id)}
