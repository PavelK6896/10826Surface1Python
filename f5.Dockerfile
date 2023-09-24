FROM python:3.10.9-slim AS dependencies1
RUN python --version
RUN pip --version
ADD requirements1.txt /requirements1.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt
ADD requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py /main.py
COPY predict.py /predict.py
COPY static /static
COPY model /model
WORKDIR /
ENV PORT 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT


# docker build --progress=plain -t m-f5 -f f5.Dockerfile .
# docker run -e PORT=8000 -p 8000:8000 --name m-f5c -d m-f5

# docker login --username oauth --password secret cr.yandex
# docker image tag m-f5 cr.yandex/crpbtkqol2ing4gt1s4p/m1:v2
# docker push cr.yandex/crpbtkqol2ing4gt1s4p/m1:v2







