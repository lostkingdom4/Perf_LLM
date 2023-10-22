export CUDA_VISIBLE_DEVICES=0
python Only_FT.py --RL_steps 20 \
					--batch_size 32 \
					--save_total_limit 1 \
					--report --per_device_train_batch_size 16 \
					--num_train_epochs 2 \
					--save_datasets \
					--load_from_model_path "/root/sk/Perf_LLM-main/model_param/model_5_2023-10-19_21-53-17.pth"
