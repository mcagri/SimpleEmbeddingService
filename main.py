from fastapi import FastAPI
from Model import ModelHandler
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/embedtext/")
async def get_text_embedding(obj: dict,):
    try:
        keys = obj.keys()
        assert "id" in keys
        assert "text" in keys
        obj["embedding"] = ModelHandler.get_embedding(obj["text"])
        return obj
    except AssertionError:
        return {"message": "invalid parameter, correct format: {id:id_val, text:text_val}"}
