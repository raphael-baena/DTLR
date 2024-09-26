CUDA_VISIBLE_DEVICES=0 python main_RIMES.py \
	--device cuda:0 \
	--new_class_embedding --smart_mapping \
	--resume_finetuning \
	--path_old_charset /home/rbaena/projects/OCR/DINO/charset_HWDB.pkl \
	--output_dir logs/DINO/test_HWDB -c config/DINO/HWDB_full.py --dataset_file HWDB --save_results \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0