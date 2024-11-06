weight_path="./logs/Copiale/checkpoint.pth"
python evaluation.py --dataset copiale --mode test --new_class_embedding  --metrics cipher --weights $weight_path --config config/Latin_CTC.py 
