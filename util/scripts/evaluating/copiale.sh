weight_path="/logs/Copiale"
python evaluation.py --dataset copiale --mode test --new_class_embedding  --metrics cipher --weights $weight_path --config config/Latin_CTC.py 