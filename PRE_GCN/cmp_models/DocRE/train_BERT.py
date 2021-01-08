from config import ConfigBert
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
from models.BBert import BBERT

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'dev_train_dev')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')


args = parser.parse_args()
model = {
    'BBERT':BBERT,
}

con = ConfigBert.ConfigBert(args)
con.set_max_epoch(100)
con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)
