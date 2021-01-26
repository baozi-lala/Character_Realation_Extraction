import json
import os
from _sha1 import sha1
import time
import pymongo

from flask import Flask, render_template
from flask import request
from werkzeug.utils import redirect
from ..PRE_GCN.src.test import PrototypeSystem

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello world !"


@app.route('/predict', methods=['GET', 'POST'])
def search():
    return render_template('predict.html')



def get_token():
    token = sha1(os.urandom(24)).hexdigest()
    return token


@app.route('/save/search_result', methods=['GET', 'POST'])
def save():
    data = json.loads(request.form.get('data'))
    text = data['text']
    PrototypeSystem=PrototypeSystem()
    test_result=PrototypeSystem.predict(text)





@app.route('/get/search_history', methods=['GET', 'POST'])
def get_search_history():
    doc = [item for item in cursor]
    doc = sorted(doc, key=lambda e: e.__getitem__('search_time'), reverse=True)
    print(doc)
    return {"success": True, "msg": "Success!!!", "data": doc}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
