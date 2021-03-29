import io
import os
import argparse
import time
import flask
import torch
import pickle

# insert root dir path to sys.path to import PrototypeSystem
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../PRE_GCN/src')))

from test import PrototypeSystem
# initialize Flask application
app = flask.Flask(__name__)

UNKNOWN_TOKEN_ID = 1


@app.route("/hello", methods=["GET"])
def index():
    return "Hello, world!"


@app.route("/", methods=["GET"])
def demo():
    return flask.render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    tic = time.time()
    resp = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        # Read text
        text = flask.request.values["text"]

        # process text
        data,preds = p.predict(text)
        nodes=[]
        for entity in data:
            node={}
            node["name"]=entity['name']
            node["role_id"]=entity['id']
            node["group"]= 1
            node["avatar"]= "static/img/back.png"
            nodes.append(node)
        links=[]
        for pair in preds:
            link={}
            link["source"]=pair[0]
            link["target"]=pair[1]
            link["relation"]=pair[2]
            link["color"]="734646"
            links.append(link)

        res={}
        res["nodes"] = nodes
        res["links"] =links
        resp["res"] =res
        resp["success"] = True

    toc = time.time()
    resp["elapse"] = toc-tic

    return flask.jsonify(resp)





if __name__ == '__main__':

    p=PrototypeSystem(remodelfile='../../PRE_GCN/results/docpre-dev-merge-v3/docred_full/')
    app.run(host='localhost', port=5001, debug = False)
