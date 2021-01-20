
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch', type=int, default=8)
inp = parser.parse_args()


config_path = './configs/docpre_basebert.yaml'
output_path = './results/docpre-dev/docred_basebert_full/'
# os.system('CUDA_VISIBLE_DEVICES=' + str(inp.gpu)+ ' python main.py --train --batch=' + str(inp.batch)+ ' --test_data=./data/DocRED/processed/dev.data'
#           ' --config_file=' + config_path+ ' --save_pred=dev --output_path=' + output_path)

with open(os.path.join(output_path, "train_finsh.ok"), 'r') as f:
    for line in f.readlines():
        input_theta = line.strip().split("\t")[1]
        break
input_theta ='0.4426703155040741'
# os.system('python ./main.py --test --batch ' + str(inp.batch)+ ' --test_data ./data/DocRED/processed/dev.data'
#           ' --config_file=' + config_path + ' --output_path=' + output_path
#           + ' --save_pred=dev_test --input_theta='+str('0.4426703155040741') + ' --remodelfile='+output_path)

os.system('CUDA_VISIBLE_DEVICES=' + str(inp.gpu)+ ' python ./main.py --test --batch ' + str(inp.batch)+ ' --test_data ./data/DocRED/processed/test.data'
          ' --config_file=' + config_path + ' --output_path=' + output_path
          + ' --save_pred=test --input_theta=' + str(input_theta) + ' --remodelfile=' + output_path)

# os.system('python ./data/convert2result.py --input_path='+ output_path
#            + 'test.errors --output_path=' + output_path + 'result.json')