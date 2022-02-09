# Attention-based Protein-ligand Interface Prediction 
## Usage
We provide classes in three modules:
* `runner_train.py`: core APIP modules 
* `APIP.dataset`: datasets used in APIP
* `APIP.model`: implementations of modules of APIP

The core modules in `APIP` are meant to be as general as possible, but you will likely have to modify `APIP.data` and `APIP.models` for your specific application, with the existing classes serving as examples.

## Training / testing
To train a model, simply run `python runner_train.py` with the following options.

```
$ python trian.py 
usage: train.py[--dataset-name NAME] [--setting SETTING] [--clu-thre N][--name NAME] 

optional arguments:
  --measure          options for dataset. choices=['IC50', 'KIKD']
  --setting   options for experimental settings PISC, PISP and PPI.   
	--clu_thre     clustering threshold. choices=['0.3', '0.4', '0.5', '0.6']
  ----maxlen N       max number of protein sequence length, default=3072
	--cnn_kernel		cnn_kernel size, type=int, default=7
  ----transformer_depth N          attention_depth, default=2
  ----transformer_hidden N    hidden dimension for attention layer, default=256
	--pair_loss, 		the choice of loss function, default='bce', choices=['bce', 'weighted_bce'])
  --attention_dropout  attention_dropout, default=0.1
  --batch_size N         batch size, default=16
  --name             name for the model

```




For example:
```
# train a model on KIKD dataset (cluster = 0.3) under PISC setting.
python runner_train.py KIKD new_compound 0.3 
```