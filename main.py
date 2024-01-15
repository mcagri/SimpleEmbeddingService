from fastapi import FastAPI
from Model import ModelHandler
import uvicorn
app = FastAPI()


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


if __name__ == '__main__':
    uvicorn.run(app, port=15001, host="0.0.0.0")
