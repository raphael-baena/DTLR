nseq=$1
train_file=$2
model_name=$3

./kenlm/build/bin/lmplz -o $nseq <./data/processed_text/$train_file >./models/$model_name.arpa --discount_fallback
./kenlm/build/bin/build_binary ./models/$model_name.arpa ./models/$model_name.binary