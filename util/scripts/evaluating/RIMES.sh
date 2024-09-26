weight_path="/logs/RIMES"
python evaluation.py --dataset RIMES --mode test  --weights $weight_path --config config/Latin_CTC.py --NMS 0.5 --TH 0.3  