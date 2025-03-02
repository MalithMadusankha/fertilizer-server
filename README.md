# investigation-fast-api

This ML project will predict the number of frauds in the city and predict the city that raises the maximum number of frauds. I have used Random Forest Regressor algorithms to train the model.

## Create virtual envirment :

``python -m venv myenv`

## ACTIVATE VENV:

`source myenv/Scripts/activate`

## Install Packages

requirement.text install: `pip install -r requirements.txt`

RUN SERVER: `uvicorn main:app --reload`

Test Api: http://localhost:8000/docs

app psw: axsrbizmfuepylid

localy deploy : lt --port 8000

create : pip freeze > requirements.txt
