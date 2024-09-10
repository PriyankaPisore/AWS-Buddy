from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
from transformers import TapexTokenizer, BartForConditionalGeneration, pipeline

chat = Blueprint("chat", __name__, url_prefix="/chat")

# tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
# model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
def apply_model(table, queries):

    
    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    answers = []

    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))

    result = ""

    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        question = '===> {}\n'.format(query)
        if predicted_agg == "NONE":
            #print('{}\n'.format(answer))
            answer_formatted = answer + '\n'
        else:
            answer_formatted = '{} > {}\n\n'.format(predicted_agg, answer)
        result += answer_formatted

    return result
    

# model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")
# tokenizer = AutoTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
# nlp = pipeline('table-question-answering', model=model, tokenizer=tokenizer)

tqa = pipeline(task="table-question-answering",model="google/tapas-large-finetuned-wtq")
dataset = pd.read_csv("./data_sheet.csv")
dataset = dataset.reset_index(drop=True)
# dataset['Number of Objects'] = dataset['Number of Objects'].astype(str)
# dataset['Bill Per Month'] = dataset['Bill Per Month'].astype(str)
dataset = dataset.astype(str)


@chat.route("/", methods=["GET", "POST"])
def display_sentence():
    return render_template("./index.html")


@chat.route("/ask", methods=["POST"])
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
