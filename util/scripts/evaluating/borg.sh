weight_path="/logs/Borg"
python evaluation.py --dataset borg --mode test --new_class_embedding  --metrics cipher --weights $weight_path --config config/Latin_CTC.py 
