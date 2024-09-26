# N gram model

## To install 

```bash
git clone https://github.com/kpu/kenlm.git


pip install https://github.com/kpu/kenlm/archive/master.zip
```
You will need to install the build-essential package if it is not installed. 
In case of a  [gcc and g++ version mismatch](https://askubuntu.com/questions/26498/how-to-choose-the-default-gcc-and-g-version).

```bash
pip install flashlight-text
```

```bash
mkdir -p build
cd build
cmake ..
make -j 4
```

Then test with 
```python
import kenlm
model = kenlm.Model('lm/test.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
```


## To train 
### Preprocessing

#### Character-level model 
If you want to train a character level model, you first need to preprocess the text data into characters seperated by spaces, while also creating a new token/symbol for the actual space between words. 

#### Word-level model
Much simpler preprocessing, by default the words will be sperated by spaces, so you will need to add a space before every punctuation so that it is considered a word. 

#### Script
Both types of processings are done in the following script 

```bash
bash preprocessing/process_training_text.sh
```

### Training 

Training is done through the script 

```bash
bash ./train_n_gram.sh n_sequence training_text.txt name_of_the_model
```

For example, to train a character level 5-gram model on the IAM dataset

```bash
bash ./train_n_gram.sh 5 IAM_train_text_char.txt IAM_5_gram_char
```