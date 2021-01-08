from config.Config import *
from models.BBert import BBERT
from models.CNN3 import *
from models.GCN import GCN
from models.LSTM import *
from models.BiLSTM import *
from models.ContextAware import *
import argparse

# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='BBERT', help='name of the model')
parser.add_argument('--save_name', type=str, default="BBERT")

parser.add_argument('--train_prefix', type=str, default='dev_train_dev')
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
con.train(model[args.model_name], args.save_name)
