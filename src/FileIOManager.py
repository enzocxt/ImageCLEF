import cPickle
import os.path

textual_data_path = '/path/to/data_6stdpt/Features/Textual/train_data.txt'
images_features_path = '/path/to/data_6stdpt/Features/Visual/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
all_image_ids_path = '/path/to/data_6stdpt/Features/scaleconcept16_ImgID.txt_Mod'

testing_textual_data_path = '/path/to/data_6stdpt/Features/Textual/test_data_students.txt'
teaser_textual_data_path = '/path/to/data_6stdpt/Teaser/scaleconcept16.teaser_dev_data_textual.scofeat.v20160212'
teaser_visual_data_path = '/path/to/data_6stdpt/Teaser/scaleconcept16.teaser_dev_data_visual_vgg16-relu7.dfeat'

output_filename = '/path/to/output/output-results.txt'
testing_img_ids_filename = '/path/to/data_6stdpt/Features/scaleconcept16.teaser.test.ImgID.txt'


def read_textual_file():
    with open(textual_data_path) as textual_file:
        for line in textual_file:
            yield line


def read_testing_textual_file():
    with open(testing_textual_data_path) as textual_file:
        for line in textual_file:
            yield line


def read_teaser_textual_file():
    with open(teaser_textual_data_path) as textual_file:
        for line in textual_file:
            yield line


def read_textual_file_words():
    with open(textual_data_path) as textual_file:
        for line in textual_file:
            yield line.split()[2::2]


def save_to_file(path, object):
    print("Pickling to file %s\n" % path )
    with open(path, 'wb') as file:
        cPickle.dump(object, file, cPickle.HIGHEST_PROTOCOL)


def load_from_file(path):
    print("Loading from file %s\n" % path)
    with open(path, 'rb') as file:
        return cPickle.load(file)


def existFile(path):
    return os.path.isfile(path)


def read_visual_file():
    with open(images_features_path, 'r') as fVisual:
        # the first line is description line
        # num of Imgs, num of Features
        description = fVisual.readline()
        for line in fVisual:
            yield line.split()


def read_teaser_visual_file():
    with open(teaser_visual_data_path, 'r') as fVisual:
        description = fVisual.readline()
        for line in fVisual:
            yield line.split()

def read_image_ids_from_filename(filename):
    with open(filename) as textual_file:
        for line in textual_file:
            yield line[:-1]
