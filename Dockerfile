FROM python:3.8.10
WORKDIR /app
RUN pip install aiohttp==3.8.1 aiosignal==1.2.0 async-timeout==4.0.2 attrs==21.4.0 certifi==2021.10.8 charset-normalizer==2.0.12 click==8.1.2 datasets==2.0.0 dill==0.3.4 filelock==3.6.0 frozenlist==1.3.0 fsspec==2022.3.0 huggingface-hub==0.5.0 idna==3.3 joblib==1.1.0 multidict==6.0.2 multiprocess==0.70.12.2 numpy==1.22.3 packaging==21.3 pandas==1.4.2 pyarrow==7.0.0 pyparsing==3.0.7 python-dateutil==2.8.2 pytz==2022.1 PyYAML==6.0 regex==2022.3.15 requests==2.27.1 responses==0.18.0 sacremoses==0.0.49 six==1.16.0 tokenizers==0.11.6 torch==1.11.0 tqdm==4.64.0 transformers==4.17.0 typing-extensions==4.1.1 urllib3==1.26.9 xxhash==3.0.0 yarl==1.7.2 wandb
COPY . .
ENTRYPOINT ["python", "train.py"]