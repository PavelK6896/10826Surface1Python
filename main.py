from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import Optional
import urllib

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


@app.post("/images")
async def create_upload_file(
        image: UploadFile = File(...)
):
    filename = f"{uuid.uuid4()}.png"
    contents = await image.read()
    d = adapt(contents)
    r = predict(d)

    with open(f"{IMAGEDIR}{filename}", "wb") as f:
        f.write(contents)

    return {"filename": filename, "r": str(r)}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
