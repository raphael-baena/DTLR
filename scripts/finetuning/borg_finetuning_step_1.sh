python finetuning.py \
	--device cuda:0 \
	--new_class_embedding \
	--output_dir logs/OCR_borg -c config/Latin_CTC.py --dataset_file borg --save_results \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
