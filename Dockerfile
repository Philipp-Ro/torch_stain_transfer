# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR .
COPY . .

#CMD ["python","code\models\Simple_U_net\main.py"]

