weight_path="/logs/READ"
python evaluation.py --dataset READ --mode test  --new_class_embedding --weights $weight_path --config config/Latin_CTC.py --NMS 0.5 --TH 0.3  
