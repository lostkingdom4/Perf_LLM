export CUDA_VISIBLE_DEVICES=0
python RL.py --num_train_epochs 3 \
			--load_from_model_path "/root/sk/Perf_LLM-main/model_param/model_0_2023-10-27_03-43-42.pth" \
			--save_strategy "no" \
			--report_to "tensorboard" \
			--RL_step 0 \
			--learning_rate 2e-5 \
			--bf16 True \
		    --weight_decay 0. \
		    --warmup_ratio 0.03 \
		    --tf32 True \
		    --lr_scheduler_type "cosine" \
		    --rrhf_weight 1 \