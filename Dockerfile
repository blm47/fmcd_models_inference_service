ARG PROJECT_KEY=PROJECT_KEY
#FROM docker.repo.cdl.sfera.test.vtb.ru/pim_mcrmc-ml-docker-lib/astra:1.8-cuda12.8-py3.11-libhunspell-poetry as final
FROM docker.repo.cdl.sfera.test.vtb.ru/pim_mcrmc-ml-docker-lib/astra:1.8-cuda12.8-py3.11-poetry as final
ENV REQUESTS_CA_BUNDLE=/root/vtb-ca-chain.pem
ARG project=project
ARG username=username
ARG password=password
ARG COMMIT_BRANCH
ARG PROJECT_KEY=projectKey
ARG BUILD_ID
ENV COMMIT_BRANCH=${COMMIT_BRANCH}

ENV BUILD_ID=${BUILD_ID}
WORKDIR /app

COPY . .

RUN chmod -R 777 /app

RUN pip install --upgrade pip
RUN pip install pandas==2.1.3
RUN pip install transformers==4.16.2
RUN pip install catboost==1.2.2
RUN pip install nvidia-nccl-cu12==2.27.3
RUN pip install nvidia-cusparselt-cu12==0.7.1
RUN pip install nvidia-cusparse-cu12==12.5.8.93
RUN pip install nvidia-cusolver-cu12==11.7.3.90
RUN pip install nvidia-cufft-cu12==11.3.3.83
RUN pip install nvidia-cublas-cu12==12.8.4.1
RUN pip install nvidia-cudnn-cu12==9.10.2.21
RUN pip install torch==2.8.0
RUN pip install traitlets==5.1.1
RUN pip install pygments==2.11.2
RUN pip install python-dateutil==2.8.2
RUN pip install fastapi==0.94.1
RUN pip install uvicorn==0.27.0
RUN pip install apscheduler==3.8.1
RUN pip install webdataset==0.1.103
RUN pip install pyarrow==12.0.0
RUN pip install requests==2.26.0
RUN pip install aiosqlite==0.17.0
RUN pip install boto3==1.40.7
RUN pip install pyspark==4.0.1
RUN pip install jedi==0.18.2
RUN pip install autopep8==1.6.0
RUN pip install ipython==7.32.0
RUN pip install confluent_kafka==2.8.0
RUN pip install autodynatrace==2.1.1
RUN pip install s3fs==2024.5.0
RUN pip install dadm-functions==0.5.5
RUN pip install psutil==5.9.8

EXPOSE 8080

#CMD export LD_LIBRARY_PATH=/usr/lib/oracle/12.1/client64/lib:$LD_LIBRARY_PATH && . /app/.venv/bin/activate && python3 main.py
CMD export LD_LIBRARY_PATH=/usr/lib/oracle/12.1/client64/lib:$LD_LIBRARY_PATH && python3 main.py
