import FileIOManager
from TextualDictionary import TextualDictionary

dictionary_prefix = '/path/to/processed_data/corpus/'
train_image_ids_filename = 'train-image-ids.dat'
test_image_ids_filename = 'test-image-ids.dat'

class DataManager(object):
    textual_dictionary = None

    textual_train_image_ids = []
    textual_test_image_ids = []

    def __init__(self, textual_dictionary):
        assert isinstance(textual_dictionary, TextualDictionary)
        self.textual_dictionary = textual_dictionary

    def count_textual_training_img_ids(self):
        '''
        Count and store textual training img ids.
        :return:
        '''

        if FileIOManager.existFile(dictionary_prefix+train_image_ids_filename):
            self.textual_train_image_ids = FileIOManager.load_from_file(dictionary_prefix+train_image_ids_filename)
            return

        self.textual_train_image_ids = []
        lines = FileIOManager.read_textual_file()
        for line in lines:
            line_words = line.split()
            self.textual_train_image_ids.append(line_words[0])

        FileIOManager.save_to_file(dictionary_prefix+train_image_ids_filename, self.textual_train_image_ids)

    def load_testing_img_ids(self):
        '''
        Load img testing ids.
        :return:
        '''

        if FileIOManager.existFile(dictionary_prefix+test_image_ids_filename):
            self.textual_test_image_ids = FileIOManager.load_from_file(dictionary_prefix+test_image_ids_filename)
            return

        lines = FileIOManager.read_image_ids_from_filename(FileIOManager.testing_img_ids_filename)
        for line in lines:
            self.textual_test_image_ids.append(line)

        print(len(self.textual_test_image_ids), len(self.textual_train_image_ids))

        FileIOManager.save_to_file(dictionary_prefix+test_image_ids_filename, self.textual_test_image_ids)
