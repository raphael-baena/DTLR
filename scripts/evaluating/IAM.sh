weight_path="./logs/IAM/checkpoint.pth"
python evaluation.py --dataset IAM --mode test  --weights $weight_path --config config/Latin_CTC.py \
--NMS 0.5 --TH 0.3  
