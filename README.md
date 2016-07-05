# Imageclef
Project for the Teaser 1 (Text illustration) of [ImageCLEF][1] challenge - in short  the task is given a text, you want to retrieve images that best illustrate the text. Since its start (in 2010), the aim of ImageCLEF competition has been the use of web data to improve image annotation. Thus, tasks such as automatic image description, localisation of different objects in an image and generating scene descriptions are of interest in the ImageCLEF context. Implemented strategy is Latent Dirichlet Allocation (LDA) on concatenated image-text pairs using Gensim library.

Made by @dmacjam, @enzocxt, @XueyingDeng in the Text Based Information Retrieval course at KU Leuven.
# Dataset description
In the Teaser 1 task you will have to retrieve the correct image given a text. The ground-truth for the training and testing data sets is assumed to be the actual image already associated with a document.

The data for this task is available at ImageCLEF website. The whole data file is 33GB, as it contains a very large set of images.

The training data consists of 300,000 image-webpage pairs. In addition, a development set of 3,000 image-webpage pairs is also provided for parameter-tuning.

The test set of 200,000 image-webpage pairs is held-out 7 days before submit. At test time, for a given text, you will have to choose the best illustration among the 200,000 available images.

All images have been pre-processed using a 16-layer Convolutional Neural Network (CNN), whose output for each image is 4096-dimensional vector corresponding to the activations of the relu7 layer of Oxford, extracted using the Berkeley Caffe library. More details can be found at [Caffe][2]. There is no need to do any image processing. Optionally, you may get acquainted with image feature extraction. Namely, to represent an image with a (usually high-dimensional) vector. Each component of such vectors can be interpreted as a “visual property.” You  an find a nice explaination of how to do it [here][3].

Training: In this task the idea is to create a cross-modal space where we can bridge the semantic gap between images and text. Given a textual snippet, you want to retrieve images that best illustrate the test. To accomplish this, there is a variety of methods that may be used. Here we suggest a few to get you started, but we would like for you to be creative.
1. Canonical Correlation Analysis
2. Latent Dirichlet Allocation (LDA) on concatenated image-text pairs.
3. Billingual Latent Dirichlet Allocation (BiLDA) on aligned image-text pairs.
4. Cross-modal semantic embedding.

##### IDs of the images in the dataset
Features/scaleconcept16\_ImgID.txt\_Mod.tar.gz
510123 image ids

##### Visual Features
Features/Visual/scaleconcept16\_data\_visual\_vgg16-relu7.dfeat.gz
The order is the same as in the \*ImgID.txt\_Mod

##### Images
Features/Visual/scaleconcept16\_images.zip

Contains thumbnails (maximum 640 pixels of either width or height) of the images in jpeg format. To avoid having too many files in a single directory (which is an issue for some types of partitions), the files are found in subdirectories named using the first two characters of the image ID, thus the paths of the files after extraction are of the form:

 ./scaleconcepts16\_images/{IID:0:2}/{IID}.jpg

##### Train data:  Text data with corresponding image id
Features/Textual/train\_data.txt

The processed text extracted from the webpages near where the images appeared. Each line corresponds to one image, having the same order as the data\_iids.txt list. 

The lines start with the image ID, followed by the number of extracted unique words and the corresponding word-score pairs. The scores were derived taking into account 1) the term frequency (TF), 2) the document object model (DOM) attributes, and 3) the word distance to the image. The scores are all integers and for each image the sum of scores is always \<=100000 (i.e. it is normalized).

##### Dev data (For fine-tuning hyper parameters)
\* DevData/TeaserTasks/scaleconcept16.teaser\_dev\_id.\*.tar.gz

- scaleconcept16.teaser\_dev.ImgID.txt
	IDs of 3339 images for the development set of both teaser tasks. Note that 2 images are near-duplicates and will thus not be used in this dataset. We have left them intact to avoid having participants re-download the visual features.

- scaleconcept16.teaser\_dev.TextID.txt
	IDs of 3337 webpage documents for the development set of teaser task. 

- scaleconcept16.teaser\_dev.ImgToTextID.txt
	IDs of 3337 images that appear on corresponding web pages.

\* DevData/TeaserTasks/scaleconcept16.teaser\_dev\_input\_documents.tar.gz

- scaleconcept16.teaser\_dev.docID.txt
	IDs of 3337 input text documents for the development set of the teaser task. 

- docs/{docID}
	 The text for each input document. 

\* DevData/TeaserTasks/scaleconcept16.teaser\_dev\_images.zip

Contains 3339 images for the development set of both teaser tasks in jpeg format.

\* DevData/TeaserTasks/scaleconcept16.teaser\_dev\_data\_textual.scofeat.\*.gz

The processed text extracted from 3337 webpages near where the images appeared.

As in the train data:  
The lines start with the image ID, followed by the number of extracted unique words and the corresponding word-score pairs. The scores were derived taking into account 1) the term frequency (TF), 2) the document object model (DOM) attributes, and 3) the word distance to the image. The scores are all integers and for each image the sum of scores is always \<=100000 (i.e. it is normalized).

# Python NLP Libraries
## NLTK
- [NLTK github][4]
- [NLTK homepage][5]
- [NLTK book][6]

## Gensim
- [Gensim homepage][7]

# Tutorials
- [LDA tutorial][8]


[1]:	http://imageclef.org/2016/annotation
[2]:	https://github.com/BVLC/caffe/wiki/Model-Zoo
[3]:	http://www.marekrei.com/blog/transforming-images-to-feature-vectors/
[4]:	https://github.com/nltk/nltk/wiki
[5]:	http://www.nltk.org/
[6]:	http://www.nltk.org/book_1ed/
[7]:	https://radimrehurek.com/gensim/index.html
[8]:	https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html