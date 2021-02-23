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
    # initialize arguments, configurations, logger and etc.
    # args = parse_args()
    # cfgs = easycls.helpers.parse_cfgs(args.config)
    # logger = easycls.helpers.init_root_logger(filename=os.path.join(
    #         'logs',
    #         f"{cfgs['model'].get('arch', 'null')}_deploy_{easycls.helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
    #     ))
    #
    # logger.info("Loading PyTorch model and starting Flask server ...")
    # logger.info("Please wait until server has fully started.")

    # device = args.device if torch.cuda.is_available() else 'cpu'
    # device = torch.device(args.device)

    # image backbone
    # backbone = models.resnetfmp101(pretrained=True)

    # load spaCy model
    # en = spacy.load('en_core_web_sm')

    # load vocabulary
    # vocabulary_path = cfgs["data"]["vocabulary"]
    # with open(vocabulary_path, "rb") as f:
    #     vocab = pickle.load(f)

    # misc configs
    # topk = cfgs['basic'].get('topk', 3)

    # load model
    # model_arch = cfgs["model"].get("arch")
    # model_kwargs = cfgs["model"].get("kwargs")
    # model = models.__dict__[model_arch](**model_kwargs if model_kwargs else {})
    # checkpoint = helpers.load_checkpoint(args.resume, map_location="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'])

    # run the Flask app
    # flask_kwargs = cfgs['basic'].get('flask')
    # logger.info(f"Loading Flask app with specs: {flask_kwargs}.")
    # app.run(**flask_kwargs if flask_kwargs else {})
    p=PrototypeSystem(remodelfile='../../PRE_GCN/src/results/docpre-dev-merge/docred_full/')
    app.run(host='0.0.0.0', port=5001, debug = False)
