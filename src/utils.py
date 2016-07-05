from gensim import matutils

# Number of maximum features to take into account.
number_of_elements = 100

def generate_corpus_for_image(features, features2id):
    '''
    Create sparse vector (feature_id, weight) from visual features.
    :param features:
    :param features2id:
    :return:
    '''
    image_corpus_line = []
    values = []
    for i in range(0, 4096):
        feature_value = float(features[i])
        if feature_value != 0:
            values.append(feature_value)
            image_corpus_line.append( (features2id[i], feature_value) )
    # Get only X top significant elements.
    image_corpus_line = [x for (y,x) in sorted(zip(values, image_corpus_line), reverse=True)][0:number_of_elements]
    #image_corpus_line = sorted(enumerate(image_corpus_line), key=lambda item: item[1], reverse=True)[0:number_of_elements]
    return matutils.unitvec(image_corpus_line)
