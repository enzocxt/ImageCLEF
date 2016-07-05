from DataManager import DataManager
from ConcatenatedCorpus import ConcatenatedCorpus
import gensim
import FileIOManager

lda_model_filename = '/media/dmacjam/Data disc/Open data/TBIR/processed_data/ldamodel/50-limited/lda-model'

class ConcatenatedLDA(object):
    '''
    Class represents LDA. Using gensim LDA implementation.
    '''
    data_manager = None
    lda_model = None
    corpus = None
    num_topics = None

    def __init__(self, data_manager, corpus):
        assert isinstance(data_manager, DataManager)
        assert isinstance(corpus,ConcatenatedCorpus )
        self.data_manager = data_manager
        self.corpus = corpus

    def train_lda(self):
        print("===============================================================")
        print("Training LDA.\n")
        if FileIOManager.existFile(lda_model_filename):
            self.lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_filename)
            self.num_topics = self.lda_model.num_topics
            return

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                         id2word=dict((v, k) for k, v in self.data_manager.textual_dictionary.word2id.items()),
                                                         num_topics= 50, update_every=1, passes=1, chunksize=8000)
        # Save the model.
        self.lda_model.save(lda_model_filename)
        self.num_topics = self.lda_model.num_topics

    def document_topics_inference(self, doc, min_probability=0.001):
        #return self.lda_model[doc]
        return self.lda_model.get_document_topics(doc, minimum_probability=min_probability)

    def document_topic_inference_chunk(self, chunk):
        (gamma, sstats) = self.lda_model.inference(chunk)
        return gamma

