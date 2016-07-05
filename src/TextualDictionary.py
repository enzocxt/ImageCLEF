from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import word_tokenize
import FileIOManager
from gensim import corpora

dictionary_prefix = '/path/to/processed_data/corpus/'
word2id_filename = 'word2id.dat'
id2word_filename = 'id2word.dat'
gensim_dictionary_filename = 'gensim-dictionary.dat'


class TextualDictionary(object):
    '''
    Class represents corpus dictionary.
    '''
    p_stemmer = PorterStemmer()
    stop = stopwords.words('english')

    textual_docs_count = 0
    unique_words_counter = 0
    word2id = dict()
    id2word = dict()

    # Index for speeding searching for an word2id
    features_names2id = []

    unique_words = set()

    gensim_dictionary = None

    def __init__(self):
        pass

    def create_custom_vocabulary(self):
        '''
        Create custom vocabulary by searching through textual training file.
        :return:
        '''
        print("===============================================================")
        print("Creating vocabulary - loading textual features.\n")
        # Generate feature names
        self._generate_visual_feature_names()

        if FileIOManager.existFile(dictionary_prefix+id2word_filename):
            self.load_vocabulary()
            return

        number_of_lines = 0
        lines = FileIOManager.read_textual_file()
        for line in lines:
            number_of_lines += 1
            # Get only words
            line_words = line.split()[2::2]
            for word in line_words:
                word = self.processWord(word.decode('utf-8'))
                if (word not in self.stop):
                    self.unique_words.add(word)
                    self.id2word[self.unique_words_counter] = word
                    self.word2id[word] = self.unique_words_counter
                    self.unique_words_counter += 1
            # Debug output
            if ((number_of_lines + 1) % 10000 == 0):
                print("Read %d" % (number_of_lines + 1))
        self.textual_docs_count = number_of_lines
        print("Number of text examples read:", number_of_lines)
        print("Number of unique words:", len(self.unique_words))
        # Delete unnecessary variables
        self.unique_words = None
        # Save vocabulary into file.
        self.save_vocabulary()

    def create_gensim_dictionary(self):
        '''
        Create vocabulary using gensim dictionary.
        :return:
        '''
        print("===============================================================")
        print("Creating vocabulary by dictionary in gensim - loading textual features.\n")

        if FileIOManager.existFile(dictionary_prefix+gensim_dictionary_filename):
            self.load_gensim_dictionary()
            return

        # Generate visual feature names
        features = [self._generate_visual_feature_names()]

        corpus = []
        lines = FileIOManager.read_textual_file_words()
        number_of_lines = 0
        for line in lines:
            line = [self.processWord(word.decode('utf-8')) for word in line]
            corpus.append(line)
            number_of_lines += 1
            # Debug output
            if ((number_of_lines + 1) % 10000 == 0):
                print("Read %d" % (number_of_lines + 1))
        dictionary = corpora.Dictionary(corpus)
        print(dictionary)
        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stop if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify() # remove gaps in id sequence after words that were removed
        print(dictionary)
        dictionary.add_documents(features)
        print(dictionary)
        self._create_feature_name2id_list()
        dictionary.save(dictionary_prefix+gensim_dictionary_filename)

    def load_gensim_dictionary(self):
        self.gensim_dictionary = corpora.Dictionary.load(dictionary_prefix+gensim_dictionary_filename)
        assert isinstance(self.gensim_dictionary, corpora.Dictionary)
        # self.id2word = self.gensim_dictionary.id2token # can be computed as: dict((v, k) for k, v in iteritems(self.word2id))
        self.word2id = self.gensim_dictionary.token2id
        self._create_feature_name2id_list()

    def save_vocabulary(self):
        FileIOManager.save_to_file(dictionary_prefix+id2word_filename,self.id2word)
        FileIOManager.save_to_file(dictionary_prefix+word2id_filename, self.word2id)

    def load_vocabulary(self):
        self.id2word = FileIOManager.load_from_file(dictionary_prefix+id2word_filename)
        self.word2id = FileIOManager.load_from_file(dictionary_prefix+word2id_filename)

    def processWord(self, word):
        # Tokenize word to delete apostrophe.
        #word = word_tokenize(word)[0] Too slow
        # Stem the word.
        return self.p_stemmer.stem(word)

    def _generate_visual_feature_names(self):
        '''
        Generate word names and ids for image features. They will have the corresponding id 0-4095.
        :return:
        '''
        print("===============================================================")
        print("Generating visual feature names to vocabulary.\n")
        feature_names = []
        for i in range(0, 4096):
            feature_name = "feature"+`(i+1)`
            self.id2word[self.unique_words_counter] = feature_name
            self.word2id[feature_name] = self.unique_words_counter
            self.unique_words_counter += 1
            feature_names.append(feature_name)
        return feature_names

    def _create_feature_name2id_list(self):
        '''
        Create index for feature names to speed up searching for their indices.
        :return:[feature_index] = id_in_corpus
        '''
        print("Creating feature2corpus_id index.")
        for i in range(0, 4096):
            feature_name = "feature"+`(i+1)`
            self.features_names2id.append(self.word2id[feature_name])
