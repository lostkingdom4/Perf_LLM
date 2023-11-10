export CUDA_VISIBLE_DEVICES=0
python RRHF.py --RL_steps 5 --batch_size 128 --save_total_limit 2 --report \
				--load_from_model_path "/home/mingxi/sk/HF/RRHF_perf/model_param/model_2_2023-10-26_22-45-30.pth"
