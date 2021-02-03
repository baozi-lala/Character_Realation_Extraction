import io
import os
import argparse
import time
from PIL import Image
import flask
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import spacy
import pickle

# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../PRE_GCN/src')))

from test import PrototypeSystem
# initialize Flask application
app = flask.Flask(__name__)



UNKNOWN_TOKEN_ID = 1
def parse_args():
    argparser = argparse.ArgumentParser(description='Forestry Security Server')
    argparser.add_argument(
        '-c',
        '--config',
        default='config.deploy.yml',
        type=str,
        metavar='PATH',
        help='configuration file path')
    argparser.add_argument(
        '-r',
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')
    argparser.add_argument(
        '-d',
        '--device',
        default='cuda',
        choices=('cpu', 'cuda'),
        type=str,
        metavar='DEVICE',
        help='computing device (default: cuda)')
    args = argparser.parse_args()
    return args


@app.route("/hello", methods=["GET"])
def index():
    return "Hello, world!"


@app.route("/", methods=["GET"])
def demo():
    return flask.render_template("demo.html")


@app.route("/predict", methods=["POST"])
def predict():
    tic = time.time()
    resp = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        # Read text
        text = flask.request.values["text"]

        # preprocess text
        data,preds = preprocess(text)
        nodes=[]
        for entity in data['entities']:
            node={}
            node["name"]=entity['name']
            node["role_id"]=entity['id']
            node["group"]= 1
            node["avatar"]= "./img/140646844806.jpg"
            nodes.append(node)
        links=[]
        for rel, pair in zip(preds, data['lables']):
            link={}
            link["source"]=pair['p1']
            link["target"]=pair['p2']
            link["relation"]=rel
            link["color"]="734646"
            links.append(link)
        resp["nodes"] = nodes
        resp["links"] =links

    toc = time.time()
    resp["elapse"] = toc-tic

    return flask.jsonify(resp)


def preprocess(text: str):
    """
    Preprocesstext for model inference

    Args: 
        text (str):

    Returns:
        textTensor (torch.Tensor)
    """
    # PIL.Image -> transformed image tensor
    imageTensor = default_transforms(image)
    imageTensor = imageTensor.unsqueeze(0)  # insert batch dimension

    # text -> tokenized LongTensor
    doc = en(text)
    token_ids = [vocab.get(token.text, UNKNOWN_TOKEN_ID) for token in doc]
    textTensor = torch.LongTensor(token_ids)
    textTensor = textTensor.unsqueeze(0)    # insert batch dimension

    return imageTensor, textTensor


if __name__ == '__main__':
    # initialize arguments, configurations, logger and etc.
    args = parse_args()
    cfgs = easycls.helpers.parse_cfgs(args.config)
    logger = easycls.helpers.init_root_logger(filename=os.path.join(
            'logs',
            f"{cfgs['model'].get('arch', 'null')}_deploy_{easycls.helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
        ))

    logger.info("Loading PyTorch model and starting Flask server ...")
    logger.info("Please wait until server has fully started.")

    device = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    # image backbone
    backbone = models.resnetfmp101(pretrained=True)

    # load spaCy model
    en = spacy.load('en_core_web_sm')

    # load vocabulary
    vocabulary_path = cfgs["data"]["vocabulary"]
    with open(vocabulary_path, "rb") as f:
        vocab = pickle.load(f)

    # misc configs
    topk = cfgs['basic'].get('topk', 3)

    # load model
    model_arch = cfgs["model"].get("arch")
    model_kwargs = cfgs["model"].get("kwargs")
    model = models.__dict__[model_arch](**model_kwargs if model_kwargs else {})
    checkpoint = helpers.load_checkpoint(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    # run the Flask app
    flask_kwargs = cfgs['basic'].get('flask')
    logger.info(f"Loading Flask app with specs: {flask_kwargs}.")
    app.run(**flask_kwargs if flask_kwargs else {})
