weight_path="/logs/HWDB"
python evaluation.py --dataset HWDB --mode test --new_class_embedding --new_label_enc --metrics chinese --weights $weight_path --config config/HWDB_full.py --unicode
