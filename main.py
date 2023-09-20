from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form, File, UploadFile, status
from fastapi.responses import RedirectResponse

import os
import uuid
from typing import Annotated
from predict import adapt, predict

app = FastAPI()

app.mount("/index", StaticFiles(directory="static", html=True), name="static")
IMAGEDIR = 'new/'


@app.get("/")
async def root():
    return RedirectResponse("/index", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/message")
async def root():
    return {"message": "Hello World"}


@app.post("/save")
async def save_file(right: Annotated[str, Form()], image: UploadFile = File(...)):
    if not os.path.exists(f"{IMAGEDIR}{right}"):
        os.makedirs(f"{IMAGEDIR}{right}")

    filename = f"{uuid.uuid4()}.png"
    right = right + '/'
    contents = await image.read()
    d = adapt(contents)
    d.save(f"{IMAGEDIR}{right}{filename}")

    return {"filename": filename, "right": right}


@app.post("/images")
async def create_upload_file(
        image: UploadFile = File(...)
):
    contents = await image.read()
    d = adapt(contents)
    r = predict(d)
    return {"result": str(r)}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
