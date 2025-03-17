# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.10.6-buster

#WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY k_means_klang k_means_klang

CMD uvicorn k_means_klang.api.fast:app --host 0.0.0.0 --port $PORT
