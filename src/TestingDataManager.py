from DataManager import DataManager
from ConcatenatedLDA import ConcatenatedLDA
import FileIOManager
import utils
import numpy as np
from scipy import linalg, spatial
from gensim import matutils, similarities


dictionary_prefix = '/path/to/processed_data/testing/'
testing_img_matrix_filename = 'img-doc-topic-matrix.dat'
testing_textual_matrix_filename = 'textual-doc-topic-matrix.dat'
index_filename = 'index/'
similarity_index_filename = 'similarity-index.dat'
all_img_ids_filename = 'all-img-ids.dat'
testing_img_ids_filename = 'testing-img-ids.dat'


class TestingDataManager(object):
    '''
    Class that loads and evaluates testing data.
    '''

    data_manager = None
    lda = None
    testing_img_document_topics = []
    testing_img_count = 0
    testing_textual_document_topics = []
    testing_textual_count = 0

    testing_img_ids = []
    #all_img_ids = dict()

    similarity_index = None

    def __init__(self, data_manager, lda):
        assert isinstance(data_manager, DataManager)
        assert isinstance(lda, ConcatenatedLDA)
        self.data_manager = data_manager
        self.lda = lda

    def create_testing_img_document_topic_matrix(self):
        print("===============================================================")
        print("Creating testing img_doc-topic matrix - loading visual features and applying LDA.\n")

        if FileIOManager.existFile(dictionary_prefix+testing_img_matrix_filename):
            self.testing_img_document_topics = FileIOManager.load_from_file(dictionary_prefix+testing_img_matrix_filename)
            #self.all_img_ids = FileIOManager.load_from_file(dictionary_prefix+all_img_ids_filename)
            self.testing_img_ids = FileIOManager.load_from_file(dictionary_prefix+testing_img_ids_filename)
            self.similarity_index = similarities.Similarity.load(dictionary_prefix+similarity_index_filename)
            self.testing_img_count = len(self.testing_img_document_topics)
            print("Testing img count is %d\n" % (self.testing_img_count))
            return

        lines = FileIOManager.read_visual_file()
        textual_train_images_length = len(self.data_manager.textual_train_image_ids)
        train_index = images_index = 0
        for line in lines:
            image_id = line[0]
            #self.all_img_ids[image_id] = images_index
            if train_index < textual_train_images_length and image_id == self.data_manager.textual_train_image_ids[
                train_index]:
                train_index += 1
            else:
                self.testing_img_ids.append(image_id)
                img_doc = utils.generate_corpus_for_image(line[1:],
                                                          self.data_manager.textual_dictionary.features_names2id)
                topic_vector = self.lda.document_topics_inference(img_doc, min_probability=0.001)
                #topic_vector = self.lda.document_topic_inference_chunk([img_doc])
                # Add into the list.
                self.testing_img_document_topics.append(topic_vector)
            images_index += 1
            # Debug print
            if ((images_index + 1) % 4000 == 0):
                print("Read visual features %d" % (images_index + 1))

        # Build similarity index #self.lda.document_topics_inference(self.img_document_topics
        self.similarity_index = similarities.Similarity(dictionary_prefix+index_filename,
                                                        self.testing_img_document_topics, num_features=self.lda.num_topics)
        self.similarity_index.save(dictionary_prefix+similarity_index_filename)

        FileIOManager.save_to_file(dictionary_prefix+testing_img_matrix_filename, self.testing_img_document_topics)
        #FileIOManager.save_to_file(dictionary_prefix+all_img_ids_filename, self.all_img_ids)
        FileIOManager.save_to_file(dictionary_prefix+testing_img_ids_filename, self.testing_img_ids)
        self.testing_img_count = len(self.testing_img_document_topics)
        assert self.testing_img_count == len(self.testing_img_ids)
        print("Testing img count is %d\n" % (self.testing_img_count))

    def create_testing_textual_document_topic_matrix(self):
        print("===============================================================")
        print("Creating testing texual_doc-topic matrix - loading teaser1 textual data and applying LDA.\n")

        if FileIOManager.existFile(dictionary_prefix+testing_textual_matrix_filename):
            self.testing_textual_document_topics = FileIOManager.load_from_file(dictionary_prefix+testing_textual_matrix_filename)
            self.testing_textual_count = len(self.testing_textual_document_topics)
            print("Testing textual count is %d\n" % (self.testing_textual_count))
            return

        number_of_lines = 0
        lines = FileIOManager.read_testing_textual_file()
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
            self.testing_textual_document_topics.append(self.lda.document_topics_inference(corpus_line, min_probability=0.001))

             # Debug output
            if ((number_of_lines + 1) % 10000 == 0):
                print("Read %d" % (number_of_lines + 1))

        FileIOManager.save_to_file(dictionary_prefix+testing_textual_matrix_filename, self.testing_textual_document_topics)
        self.testing_textual_count = len(self.testing_textual_document_topics)
        print("Testing textual count is %d\n" % (self.testing_textual_count))

    def yield_testing_textual_document_topics(self):
        number_of_lines = 0
        lines = FileIOManager.read_testing_textual_file()
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
            corpus_line = self.lda.document_topics_inference(corpus_line, min_probability=0.001)

            yield corpus_line



    def create_ranking(self):
        print("===============================================================")
        print("Creating ranking.\n")
        output_file = open(FileIOManager.output_filename, 'w')
        not_enough_results_counter = 0

        self.similarity_index.num_best = 100
        queries = self.yield_testing_textual_document_topics()
        for i, query in enumerate(queries):
        #for i, query in enumerate(self.testing_textual_document_topics):
            sims = self.similarity_index[query]
            sim_length = len(sims)
            if sim_length < 100:
                not_enough_results_counter += 1
            for j in range(0, 100):
                if j < sim_length:
                    id, weight = sims[j]
                else:
                    # if no value, let id be something
                    id, weight = not_enough_results_counter, 0
                # Lookup image_id in dictionary by image name
                image_id = self.testing_img_ids[id]             #self.all_img_ids.index[self.testing_img_ids[id]]
                output_file.write("%d 0 %s 0 %.2f 0\n" % (i, image_id, weight ))

            # Debug output
            if ((i + 1) % 10000 == 0):
                print("Written %d" % (i + 1))

        output_file.close()
        print("Not enough results for %d documents" % not_enough_results_counter)


