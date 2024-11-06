weight_path="./logs/Borg/checkpoint.pth"
python evaluation.py --dataset borg --mode test --new_class_embedding  --metrics cipher --weights $weight_path --config config/Latin_CTC.py 
