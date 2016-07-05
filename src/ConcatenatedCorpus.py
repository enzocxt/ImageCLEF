from TextualDictionary import TextualDictionary
import FileIOManager
import utils
from gensim import matutils


class ConcatenatedCorpus(object):
    '''
    Class representing training corpus with concatenated (txt, img) features. Implemented as lazy loading.
    '''
    length = 0
    dictionary = None
    limited_length = None

    def __init__(self, length, dictionary):
        assert isinstance(dictionary, TextualDictionary)
        self.length = length
        self.dictionary = dictionary

    def __iter__(self):
        textual_lines = FileIOManager.read_textual_file()
        visual_file = open(FileIOManager.images_features_path, 'r')
        visual_file.readline()
        number_of_lines = 0
        for textual_line in textual_lines:
            number_of_lines += 1
            if self.limited_length is not None and number_of_lines > self.limited_length:
                break

            corpus_line_dict = dict()
            line_words = textual_line.split()
            textual_img_id = line_words[0]
            number_of_features = int(line_words[1])
            line_words = line_words[2:]
            for j in range(0, number_of_features*2, 2):
                word = self.dictionary.processWord(line_words[j].decode('utf-8'))
                # Normalize weight
                weight = float(line_words[j + 1]) / 100000
                if word not in self.dictionary.word2id:
                    continue
                # Get word id
                word_id = self.dictionary.word2id[word]
                if word_id not in corpus_line_dict:
                    corpus_line_dict[word_id] = weight
                else:
                    corpus_line_dict[word_id] += weight

            # Create array of tuples (word_id, weight) from dictionary
            corpus_line = []
            for key, value in corpus_line_dict.iteritems():
                corpus_line.append( (key, value) )

            # Normalize to unit vector
            corpus_line = matutils.unitvec(corpus_line)

            # Search for training images only for corresponding img
            visual_line = visual_file.readline().split()
            image_id = visual_line[0]
            while image_id != textual_img_id:
                visual_line = visual_file.readline().split()
                image_id = visual_line[0]

            # Append visual features
            corpus_line = corpus_line + utils.generate_corpus_for_image(visual_line[1:], self.dictionary.features_names2id)

            # Normalize to unit vector
            corpus_line = matutils.unitvec(corpus_line)

            yield corpus_line

    def __len__(self):
        if self.limited_length is not None:
            return self.limited_length
        else:
            return self.length
