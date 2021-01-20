
from config.Config import *
# import models
from models.BBert import BBERT
from models.CNN3 import CNN3
from models.GCN import GCN
from models.LSTM import *
from models.BiLSTM import *
from models.ContextAware import *
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='BiLSTM', help='name of the model')
parser.add_argument('--save_name', type=str, default="BiLSTM")

parser.add_argument('--train_prefix', type=str, default='dev_train')
parser.add_argument('--test_prefix', type=str, default='dev_dev')

args = parser.parse_args()
model = {
    'CNN3': CNN3,
    'LSTM': LSTM,
    'BiLSTM': BiLSTM,
    'ContextAware': ContextAware,
    'BBERT':BBERT
}

con = Config(args)
con.set_max_epoch(86)
con.load_train_data()
con.load_test_data()
# con.set_train_model()
model_name=['CNN3','BiLSTM','LSTM','ContextAware']
for name in model_name:
    print("---------",name,"------------")
    con.train(model[name], args.save_name)
    print("over")
