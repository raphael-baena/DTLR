mkdir ./data/processed_text
mkdir ./models/

echo "Saving training data to txt file from labels.pkl"

python preprocessing/get_training_text.py --dataset IAM

echo "#####"
python preprocessing/get_training_text.py --dataset READ
echo "#####"
python preprocessing/get_training_text.py --dataset RIMES


echo "Processing text to character level format"

python preprocessing/get_char_training_text.py --dataset IAM
echo "#####"

python preprocessing/get_char_training_text.py --dataset READ
echo "#####"

python preprocessing/get_char_training_text.py --dataset RIMES
