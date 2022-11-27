from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

##################### Load Model #########################
def load_model():
    return joblib.load("model/iris_classifier.joblib")

####################### EndPoints ##########################

@app.post("/predict/")
# async def create_item(input: InferenceParameters, user_id: int):
async def inference(input: list):
    clf = load_model()

    df_input = pd.DataFrame([input])

    result = clf.predict(df_input)
    print(result)
    return{"whole_input": input, "result": float(result[0]) }

@app.get("/")
async def root():
    model = load_model()
    return {"message": "OK"}