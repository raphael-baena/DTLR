
python main_synthetic.py \
	--device cuda:0 \
	--output_dir logs/OCR_general -c config/Latin.py --dataset_file synthetic_line_OCR_general --save_results \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

