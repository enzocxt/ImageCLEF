from DataManager import DataManager
from ConcatenatedLDA import ConcatenatedLDA
import FileIOManager
import utils
import numpy as np
from gensim import matutils, similarities
from scipy import linalg, spatial

dictionary_prefix = '/path/to/processed_data/teaser/'
img_matrix_filename = 'img-topic-matrix.dat'
textual_matrix_filename = 'txt-topic-matrix.dat'
similarity_index_filename = 'sim-index.dat'
index_filename = 'index/'

textual_image_ids_filename = '/path/to/data_6stdpt/Teaser/scaleconcept16.teaser_dev.TextID.txt'
visual_image_ids_filename = '/path/to/data_6stdpt/Teaser/scaleconcept16.teaser_dev.ImgID.txt'
ground_truth_filename = '/path/to/data_6stdpt/Teaser/scaleconcept16.teaser_dev.ImgToTextID.txt'


class TeaserDataManager(object):
    '''
    Class that load and test teaser data.
    '''

    data_manager = None
    lda = None

    img_document_topics = []
    visual_image_ids = []

    textual_document_topics = []
    textual_image_ids = []

    query2img = dict()

    similarity_index = None

    def __init__(self, data_manager, lda):
        assert isinstance(data_manager, DataManager)
        assert isinstance(lda, ConcatenatedLDA)
        self.data_manager = data_manager
        self.lda = lda
        self.read_ground_truth()

    def create_img_document_topic_matrix(self):
        print("===============================================================")
        print("Creating teaser img_doc-topic matrix - loading visual features and applying LDA.\n")

        self.read_visual_image_ids()

        if FileIOManager.existFile(dictionary_prefix+img_matrix_filename):
            self.img_document_topics = FileIOManager.load_from_file(dictionary_prefix+img_matrix_filename)
            self.similarity_index = similarities.Similarity.load(dictionary_prefix+similarity_index_filename)
            return

        lines = FileIOManager.read_teaser_visual_file()
        for line in lines:
            img_doc = utils.generate_corpus_for_image(line,
                                                          self.data_manager.textual_dictionary.features_names2id)
            topic_vector = self.lda.document_topics_inference(img_doc, min_probability=0.0001)
            # Add into the list.
            self.img_document_topics.append(topic_vector)

        # Build similarity index #self.lda.document_topics_inference(self.img_document_topics
        self.similarity_index = similarities.Similarity(dictionary_prefix+index_filename, self.img_document_topics,
                                                        num_features=self.lda.num_topics)
        self.similarity_index.save(dictionary_prefix+similarity_index_filename)
        FileIOManager.save_to_file(dictionary_prefix+img_matrix_filename, self.img_document_topics)
        testing_img_count = len(self.img_document_topics)
        print("Testing img count is %d\n" % (testing_img_count))


    def read_visual_image_ids(self):
        with open(visual_image_ids_filename) as textual_file:
            for line in textual_file:
                self.visual_image_ids.append(line[:-1])

    def create_textual_document_topic_matrix(self):
        print("===============================================================")
        print("Creating testing texual_doc-topic matrix - loading teaser1 textual data and applying LDA.\n")

        self.read_textual_image_ids()

        if FileIOManager.existFile(dictionary_prefix+textual_matrix_filename):
            self.textual_document_topics = FileIOManager.load_from_file(dictionary_prefix+textual_matrix_filename)
            print len(self.textual_document_topics)
            return

        number_of_lines = 0
        lines = FileIOManager.read_teaser_textual_file()
        for line in lines:
            number_of_lines += 1
            corpus_line_dict = dict()
            line_words = line.split()
            number_of_features = int(line_words[1])
            line_words = line_words[2:]
            for j in range(0, number_of_features*2, 2):
                word = self.data_manager.textual_dictionary.processWord(line_words[j].decode('utf-8'))
                # Normalize weight
                weight = float(line_words[j + 1]) / 100000
                if word not in self.data_manager.textual_dictionary.word2id:
                    continue
                # Get word id
                word_id = self.data_manager.textual_dictionary.word2id[word]
                if word_id not in corpus_line_dict:
                    corpus_line_dict[word_id] = weight
                else:
                    corpus_line_dict[word_id] += weight
            # Create array of tuples (word_id, weight) from dictionary
            corpus_line = []
            for key, value in corpus_line_dict.iteritems():
                corpus_line.append( (key, value) )

            # Normalize to one vector
            corpus_line = matutils.unitvec(corpus_line)

            # Add into the list.
            self.textual_document_topics.append(self.lda.document_topics_inference(corpus_line, min_probability=0.001))   # self.lda.document_topics_inference(

        FileIOManager.save_to_file(dictionary_prefix+textual_matrix_filename, self.textual_document_topics)
        testing_textual_count = len(self.textual_document_topics)
        print("Testing textual count is %d\n" % (testing_textual_count))

    '''
    def create_word_topic_matrix(self, query, num_topics=150):
        query_words = []
        text_words = query.split()
        textID, numWords = text_words[0], int(text_words[1])
        for j in range(0, numWords*2, 2):
            word = self.data_manager.textual_dictionary.processWord(text_words[j].decode('utf-8'))
            if word not in self.data_manager.textual_dictionary.word2id:
                    continue
            # Get word id
            word_id = self.data_manager.textual_dictionary.word2id[word]
            query_words.append(word)

        num_words = len(query_words)
        wordTopic_Matrix = np.zeros((num_words, num_topics), dtype='float32')
        total_number_of_words = len(self.data_manager.textual_dictionary.gensim_dictionary.keys())

        for i in range(0, num_words):
            for j in range(0, num_topics):
                word_distr = self.lda.lda_model.get_topic_terms(j, topn=200)
                for wordId, p in word_distr:
                    if self.data_manager.textual_dictionary.word2id[query_words[i]] == wordId:
                        wordTopic_Matrix[i][j] = p
        print wordTopic_Matrix
        '''

    def read_textual_image_ids(self):
        with open(textual_image_ids_filename) as textual_file:
            for line in textual_file:
                self.textual_image_ids.append(line[:-1])

    def read_ground_truth(self):
        print("===============================================================")
        print("Reading teaser ground truth.\n")
        with open(ground_truth_filename) as textual_file:
            for line in textual_file:
                line = line.split()
                self.query2img[line[1]] = line[0]

    def create_ranking(self):
        print("===============================================================")
        print("Creating ranking.\n")

        self.similarity_index.num_best = 100
        score = 0
        not_enough_results_counter = 0
        for i, query in enumerate(self.textual_document_topics):
            sims = self.similarity_index[query]
            #print("Query #%s" % (i+1))
            ground_truth = self.query2img[self.textual_image_ids[i]]
            ground_truth_id = self.visual_image_ids.index(ground_truth)
            sim_length = len(sims)
            if sim_length < 100:
                not_enough_results_counter += 1
            for j in range(0, sim_length):
                id, weight = sims[j]
                #print("Query #%s\timg_id:%s\ttruth:%s\tscore:%s" % (i, id, ground_truth_id, weight))
                if id == ground_truth_id:
                    print("Position %d with weight %s" % (j, weight))
                    score += 1
            #raw_input("Press Enter to continue...")

        print("Total score: %d" % score)
        print("Not enough results for %d documents" % not_enough_results_counter)

    def create_hellinger_ranking(self):
        '''
        Compute hellinger distance in matrix form. Worse results than cosine similarity. Why?
        '''
        print len(self.img_document_topics), len(self.textual_document_topics)

        # For the input need only sparse corpus, not with applied lda.
        gamma_img = self.lda.document_topic_inference_chunk(self.img_document_topics)
        gamma_txt = self.lda.document_topic_inference_chunk(self.textual_document_topics)

        # TODO process in chunks
        #results = np.dot(gamma_txt, np.transpose(gamma_img)) / linalg.norm(gamma_img)/ linalg.norm(gamma_txt)
        results = spatial.distance.cdist(gamma_txt, gamma_img, lambda u, v: np.sqrt(0.5 * (np.sqrt(u) - np.sqrt(v))**2).sum())
        score = 0
        for i, result in enumerate(results):
            max_indices = np.argsort(result)[::-1]
            assert len(max_indices) == len(self.visual_image_ids)
            ground_truth = self.query2img[self.textual_image_ids[i]]
            ground_truth_id = self.visual_image_ids.index(ground_truth)
            #predicted_img_ids = self.visual_image_ids[max_indices]
            results_weights = result[max_indices]
            for j in range(0, 100):
                #print("Query #%s\timg_id:%s\ttruth:%s\tscore:%s" % (i, max_indices[j], ground_truth_id, results_weights[j]))
                if max_indices[j] == ground_truth_id:
                    print("Position %d with weight %s" % (j, results_weights[j]))
                    score += 1
            #raw_input("Press Enter to continue...")
        print("Total score: %d" % score)

    '''
    def predict(self):
        print("===============================================================")
        print("Predicting.\n")
        lines = FileIOManager.read_teaser_textual_file()
        for line in lines:
            self.create_word_topic_matrix(line, num_topics=150)
    '''


