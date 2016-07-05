#Working code for TBIR 
#KU Leuven CS 2016
#@author: DENG-CHENG-MACINA


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings


textFileAddress= '/home/ubuntu/questions-words.txt'
gloveAddress_50D='/home/ubuntu/glove.6B.50d.txt'
gloveAddress_100D='/home/ubuntu/glove.6B.100d.txt'
gloveAddress_300D='/home/ubuntu/glove.6B.300d.txt'
output='/home/ubuntu/out.txt'
warnings.filterwarnings("ignore")

#loadvecfiles with amount
def loadGloveFile(vecFile,number_of_examples_to_load):
    with open(vecFile,'r') as textual_file:
        for i in range(0, number_of_examples_to_load):
            yield textual_file.readline()

#load vecfile & question file
def loadfile(vecFile):
    global vectorDictionary
    global wordDictionary
    global vecMatrix
    global lineText
    vectorDictionary = {}
    wordDictionary = {}
    vectorList = []
    vecMatrix = []
    linevec = []
    linevec =loadGloveFile(vecFile,30000)
    for i, linev in enumerate(linevec):
        vec = map(float, linev.split()[1:])
        vectorDictionary[linev.split()[0]] = vec
        wordDictionary[i] = linev.split()[0]
        vectorList.append(vec)
    vecMatrix = np.matrix(vectorList)
    txtFile = open(textFileAddress, 'r')
    global  lineText
    lineText = [linet.split() for linet in txtFile.readlines() if ":" not in linet.split()]


#addtion function
def gloveSolver_addition ():
    count = 0
    global vectorDictionary
    global wordDictionary
    for i, linet in enumerate(lineText):
        if vectorDictionary.has_key(str(linet[0]).lower()):
            a = np.array(vectorDictionary[str(linet[0]).lower()])
        if vectorDictionary.has_key(str(linet[1]).lower()):
            b = np.array(vectorDictionary[str(linet[1]).lower()])
        if vectorDictionary.has_key(str(linet[2]).lower()):
            c = np.array(vectorDictionary[str(linet[2]).lower()])

            # calculate the result of argmax cosin(c-a+b,d')

            similarity = cosine_similarity((c-a+b).reshape(1,-1),vecMatrix[0:30000])
            indexOfSimilarity = np.argmax(similarity)

            if wordDictionary[indexOfSimilarity] == str(linet[3]).lower():
                count = count + 1
                # print f, count
    additionModal = float(count)/float(len(lineText))
    print f, additionModal


#direction_function
def gloveSolver_direction( ):
    count = 0
    for i, linet in enumerate(lineText):
        # linet = str(linet).lower()
        word = str(linet[2]).lower()
        if vectorDictionary.has_key(str(linet[0]).lower()):
            a = np.array(vectorDictionary[str(linet[0]).lower()])
        if vectorDictionary.has_key(str(linet[1]).lower()):
            b = np.array(vectorDictionary[str(linet[1]).lower()])
        if vectorDictionary.has_key(str(linet[2]).lower()):
            c = np.array(vectorDictionary[str(linet[2]).lower()])

            #calculate the result of argmax cosin(a-b,c-d)

            similarity = cosine_similarity((a-b).reshape(1,-1), c-vecMatrix[0:30000])
            indexOfSimilarity = np.argmax(similarity)

            if wordDictionary[indexOfSimilarity] == str(linet[3]).lower():
                count = count + 1

    directionModal = float(count)/float(len(lineText))
    print f, directionModal


#multiplication function
def gloveSolver_multiplication( ):
    count = 0
    global deviation
    deviation = 0.001
    for i, linet in enumerate(lineText):
        # linet = str(linet).lower()
        word = str(linet[2]).lower()
        if vectorDictionary.has_key(str(linet[0]).lower()):
            a = np.array(vectorDictionary[str(linet[0]).lower()])
        if vectorDictionary.has_key(str(linet[1]).lower()):
            b = np.array(vectorDictionary[str(linet[1]).lower()])
        if vectorDictionary.has_key(str(linet[2]).lower()):
            c = np.array(vectorDictionary[str(linet[2]).lower()])
            aresult=((cosine_similarity(a,vecMatrix[0:30000]))+1)/2
            bresult=((cosine_similarity(b,vecMatrix[0:30000]))+1)/2
            cresult=((cosine_similarity(c, vecMatrix[0:30000]))+1) / 2
            if aresult.all == 0:
                aresult=aresult+deviation
            similarity = cresult * bresult/aresult;
            indexOfSimilarity = np.argmax(similarity)
            if wordDictionary[indexOfSimilarity] ==str(linet[3]).lower():
                count = count + 1

    multiplicationModal = float(count)/float(len(lineText))
    print f,  multiplicationModal


#
#Deal with out put
#

if __name__ == '__main__':
    f = open(output, 'w')


    #this part outputs the result of 50 dimension vectors.

    loadfile(gloveAddress_50D)
    print f, "result of addition modal with 50 dimension vector is:"
    gloveSolver_addition()
    print f, "result of direction modal with 50 dimension vector is:"
    gloveSolver_direction()
    print f, "result of multiplication modal with 50 dimension vector is:"
    gloveSolver_multiplication()


    del lineText                #free the memory to load new file
    loadfile(gloveAddress_100D)

    # this part outputs the result of 100 dimension vectors.

    print f, "result of addition modal with 100 dimension vector is:"
    gloveSolver_addition()
    print f, "result of direction modal with 100 dimension vector is:"
    gloveSolver_direction()
    print f, "result of multiplication modal with 100 dimension vector is:"
    gloveSolver_multiplication()

    del lineText
    loadfile(gloveAddress_300D)

    # this part outputs the result of 300 dimension vectors.

    gloveSolver_addition()
    print f,  "result of direction modal with 300 dimension vector is:"
    gloveSolver_direction()
    print f, "result of multiplication modal with 100 dimension vector is:"
    gloveSolver_multiplication()





