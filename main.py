import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import find_better_model,predict
from datetime import datetime

app = FastAPI(
    title="Iris Predictor",
    docs_url="/"
)

app.add_event_handler("startup", find_better_model)

class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class QueryOut(BaseModel):
    flower_class: str
    timestamp_str:str

class FeedbackIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class:int


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower( query_data: QueryIn):
    result = predict(query_data)
    ct     = datetime.now()
    ctStr  = ct.strftime("%m/%d/%Y, %H:%M:%S")
    output = {'flower_class': result,'timestamp_str':ctStr}
    return output


## added this from session and there is no response here except string
# @app.post("/feedback_loop", status_code=200)
# def feedback_loop( data:list[FeedbackIn]):
#     retrain(data)
#     return {"detail": "Feed back loop successful"}

# @app.post("/feedback_single_data", status_code=200)
# def feedback_single_data( data:FeedbackIn):
#     retrain_single(data)
#     return {"detail": "Feed single data successful"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8888, reload=True)
