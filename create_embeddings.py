from gensim.models import Word2Vec
from util import load_data_and_labels_from_csv_file, process_text
import os
import requests # This library is used to make requests to internet
import zipfile

from gensim.models.callbacks import CallbackAny2Vec
'''
class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words

    def on_epoch_end(self, model):
        print("Model loss:", model.get_latest_training_loss())  # print loss
        #for word in self._test_words:  # show wv logic changes
        #    print(word, "\t", model.wv.most_similar(word))
'''

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

data_file = "data/SMSSpamCollection"

#Download and unzip the data file in data directory in case it doesn't exists already
if not os.path.exists(data_file):
    data_file_dir = os.path.dirname(data_file)
    if not os.path.exists(data_file_dir): os.makedirs(data_file_dir)

    # We are storing url of dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(url, allow_redirects=True)
    zip_file_download = data_file_dir + '/smsspamcollection.zip'

    # We are writing the content of above request to 'iris.data' file
    open(zip_file_download, 'wb').write(r.content)

    #Extract the zip file
    with zipfile.ZipFile(zip_file_download,"r") as zip_ref:
        zip_ref.extractall(data_file_dir)

# Load data
print("Loading data...")
labels, sentences = load_data_and_labels_from_csv_file(data_file)

word2vec_feed = [(process_text(sent)).split(" ") for sent in sentences]

print(word2vec_feed[:5])
print('length word2vec_feed : ', len(word2vec_feed))

#monitor = MonitorCallback(["hours", "story", "sis"])  # monitor with demo words
monitor = callback()
#sg=0 for CBOW
model = Word2Vec(word2vec_feed, size=300, window=10, min_count=1, workers=24, negative=15, hs=0, sample=0.00001, sg=0, compute_loss=True, iter=1000,  callbacks=[monitor])
#model = Word2Vec(word2vec_feed, vector_size=300, window=10, min_count=1, workers=24, negative=15, hs=0, sample=0.00001, sg=0, compute_loss=True, callbacks=[monitor])

#model.train(word2vec_feed, total_examples=model.corpus_count, epochs=1000, report_delay=1)

print('word2vec model created')
print('saving the word2vec model')

model.save("sms-data-embeddings.model")
model.wv.save_word2vec_format("sms-data-embeddings.bin",  binary=True)

print('size of vocab : ' , len(model.wv.vocab))
#words = list(model.wv.index_to_key)
#print('size of vocab : ' , words)
print('process complete')

#word_vectors = model.wv
#result = word_vectors.similar_by_word("hours")
#print(result[:100])
