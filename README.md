Example of creating Word2vec embeddings using gensim

Dataset Link: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Dataset References: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

Create the data directory: 

mkdir data

Downlad the data file zip into data directory:

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip -P data/

cd data

unzip smsspamcollection.zip 

you get your data file SMSSpamCollection in data directory

Reference:
- https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
- https://radimrehurek.com/gensim/models/word2vec.html

Usage:

python3 create_embeddings.py

It will create the following two embedding files:
- sms-data-embeddings.model
- sms-data-embeddings.bin

