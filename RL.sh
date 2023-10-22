export CUDA_VISIBLE_DEVICES=0
python RRHF.py --RL_steps 5 --batch_size 128 --save_total_limit 2 --report \
				--load_from_model_path "/root/sk/Perf_LLM-main/model_param/model_auto.pth"
