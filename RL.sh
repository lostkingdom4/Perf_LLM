export CUDA_VISIBLE_DEVICES=0
python RRHF.py --RL_steps 4 --batch_size 32 --report --save_total_limit 5
