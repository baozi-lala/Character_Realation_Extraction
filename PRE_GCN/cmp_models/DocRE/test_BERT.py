from config import ConfigBert
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
from models.BBert import BBERT

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'LSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = -1)
# parser.add_argument('--ignore_input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
    'BBERT':BBERT,
}

con = ConfigBert.ConfigBert(args)
#con.load_train_data()
con.load_test_data()
# con.set_train_model()

pretrain_model_name = 'checkpoint_BiLSTM_bert_relation_exist_cls'
con.testall(model[args.model_name], args.save_name, args.input_theta, args.two_phase, pretrain_model_name)#, args.ignore_input_theta)
