from TextualDictionary import TextualDictionary
from DataManager import DataManager
from ConcatenatedCorpus import ConcatenatedCorpus
from ConcatenatedLDA import ConcatenatedLDA
from TestingDataManager import TestingDataManager
from TeaserDataManager import TeaserDataManager
import logging

# Set up gensim logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# teaser data
def teaser(data_manager, lda):
    teaser_data_manager = TeaserDataManager(data_manager, lda)
    teaser_data_manager.create_img_document_topic_matrix()
    teaser_data_manager.create_textual_document_topic_matrix()
    teaser_data_manager.create_ranking()

    #teaser_data_manager.create_hellinger_ranking()
    #teaser_data_manager.predict()

# testing data
def test(data_manager, lda):
    testing_data_manager = TestingDataManager(data_manager, lda)
    testing_data_manager.create_testing_img_document_topic_matrix()
    testing_data_manager.create_testing_textual_document_topic_matrix()
    testing_data_manager.create_ranking()


if __name__ == '__main__':

    # Create vocabulary - either custom or by gensim
    textual_dictionary = TextualDictionary()
    #textual_dictionary.create_custom_vocabulary()
    textual_dictionary.create_gensim_dictionary()

    data_manager = DataManager(textual_dictionary)
    data_manager.count_textual_training_img_ids()
    #data_manager.load_testing_img_ids()

    # Inicialize text corpus for training.
    corpus = ConcatenatedCorpus(len(data_manager.textual_train_image_ids), textual_dictionary)

    # Inicilize LDA and start training.
    lda = ConcatenatedLDA(data_manager, corpus)
    lda.train_lda()

    teaser(data_manager, lda)



