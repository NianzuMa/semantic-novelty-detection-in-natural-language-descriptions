# README

## Installation Instruction

### install transformers
It is important that transformers==3.1.0, otherwise the program will report some error

* create conda virtual environment `conda create -n gat_reason python=3.7`


install pytorch 1.6 and cuda 10, the problem is fixed.


### install pytorch 1.6 version

#### CUDA 10.2
```conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch```

### install pytorch 1.6's geometric
```
export CUDA=cu102
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric
```

pip install --no-index torch-sparse with '--no-index' parameter does not work
try go without --no-index

### update the wordnet version

1. using nltk.download() to check where the downloaded wordnet is stored.
2. download version 3.1 in wordnet wn3.1.dict.tar.gz
3. unzip and replace all files in the folder of local wordnet


### run

bash run.sh

The first run of the model take very long time, like 2 hours, since the code will need to generate cache files.
The cache files are entity similarity list computed from WordNet. These cache files are in total around 15 GB.
