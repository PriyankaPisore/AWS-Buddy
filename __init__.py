from flask import Flask, request, jsonify
import os
from flask_cors import CORS, cross_origin
from pyngrok import ngrok
import threading
from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
from transformers import TapexTokenizer, BartForConditionalGeneration, pipeline

# from flask_caching import Cache
os.environ['FLASK_ENV'] = "development"


#def create_app(test_config=None):
    # pyngrok = PyNgrok()
    # pymysql.install_as_MySQLdb()

app = Flask(__name__, instance_relative_config=True)
CORS(app)
port=5000
ngrok.set_auth_token("2clJ8ZLAFYkVM4o0OdRokc7FJfV_UnbPAbRQ8Z19R9Rn1sBn")
public_url = ngrok.connect(port).public_url
app.config['BASE_URL']=public_url

print(f"url is {public_url} and port is {port}")

# cache = Cache(app)
# cache.clear()
threading.Thread(target=app.run,kwargs={"use_reloader": False}).start()
# return app

tqa = pipeline(task="table-question-answering",model="google/tapas-large-finetuned-wtq")
dataset = pd.read_csv("./data_sheet.csv")
dataset = dataset.reset_index(drop=True)
# dataset['Number of Objects'] = dataset['Number of Objects'].astype(str)
# dataset['Bill Per Month'] = dataset['Bill Per Month'].astype(str)
dataset = dataset.astype(str)




@app.route("/chat", methods=["GET", "POST"])
def display_sentence():
    return render_template("./index.html")


@app.route("/chat/ask", methods=["POST"])
def ask():
    # chat_question = request.form['chat_box_input']
    request_parameter = request.get_json(force=True)
    chat_question = request_parameter["chat_box_input"]
    # response = dl.chat(chat_question)
    # return response


    # encoding = tokenizer(table=dataset, query=chat_question,return_tensors="pt")
    # outputs = model.generate(**encoding)
    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # return(result)
    # return(apply_model(dataset, [chat_question]))

    return (tqa(table=dataset,query=chat_question))
