dataset_path="/home/rbaena/datasets"
old_charset_path="$dataset_path/HWDB_v1/charset.pkl"

python finetuning.py \
	--device cuda:0 \
	--new_class_embedding --smart_mapping \
	--output_dir logs/HWDB -c config/HWDB_full.py --dataset_file HWDB --save_results \
	--path_old_charset "$old_charset_path" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
