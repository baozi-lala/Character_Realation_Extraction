CUDA_VISIBLE_DEVICES=1 python main.py --model MLRGNN --train --dataset docred --local_rep_att_single 1 --new_flag 1 --context_att 1 --att_head_num 2 --pretrain_l_m none --batch 4

CUDA_VISIBLE_DEVICES=1 python main.py --model EOG --test --remodelfile ../results/docred-dev/EOG_50.91 --input_theta 0.3693

CUDA_VISIBLE_DEVICES=1 python main.py --model CGCN --test --remodelfile ../results/docred-dev/CGCN_0.5214 --input_theta 0.7794
