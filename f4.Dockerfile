FROM python:3.10.9-slim AS dependencies1
RUN python --version
RUN pip --version
ADD requirements1.txt /requirements1.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt

FROM  dependencies1 AS dependencies2
RUN python --version
RUN pip --version
ADD requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

FROM dependencies2 AS build
COPY . /
WORKDIR /
ENV PORT 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
# --log-level critical

# docker build --progress=plain -t m-f4 -f f4.Dockerfile .
# docker run -e PORT=8000 -p 8000:8000 --name m-f41c -d m-f4

# docker login --username oauth --password secret cr.yandex
# docker image tag m-f4 cr.yandex/crpbtkqol2ing4gt1s4p/m1:v1
# docker push cr.yandex/crpbtkqol2ing4gt1s4p/m1:v1







