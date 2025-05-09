python finetuning.py \
	--device cuda:0 \
	--new_class_embedding --smart_mapping \
	--random_erasing \
	--resume_finetuning \
	--output_dir logs/IAM -c config/Latin_CTC.py --dataset_file IAM --save_results \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
