#!/usr/bin/python
from scipy import spatial
import random

question_path = './questions-words.txt'
word2vec_model = '/Users/enzo/Desktop/GoogleNews-vectors-negative300.txt'

count = 0
analogies = []
words = {}
# fetch the analogies
with open(question_path, 'r') as fq:
    for analogy in fq.readlines():
        if ':' in analogy: continue
        else:
            analogy.strip('\n')
            tmp = analogy.split()
            analogies.append(tmp)
            for w in tmp: words[w] = 1

print "Number of analogies: %s" % len(analogies)

vectors = [[None, None, None, None] for i in range(len(analogies))]
checked = 0
#"""
with open(word2vec_model, 'r') as fvec:
    description = fvec.readline()
    nWords, nFeatures, tmp1, tmp2 = description.split()
    print "%s words, %s features" % (nWords, nFeatures)
    count = 0
    for line in fvec:
            count += 1
            if count%10000 == 0: print count, checked, 3*len(analogies)
            if checked == 3*len(analogies): break
            line.strip('\n')
            wv = line.split()
            word, vector = wv[0], wv[1:]
            if not words.has_key(word): continue
            vector = [float(vector[i]) for i in range(len(vector))]
            for i in range(len(analogies)):
                for j in range(3):
                    if word == analogies[i][j]:
                        vectors[i][j] = vector
                        checked += 1
    print count, checked
#"""

def addition():
    print "Addition method"
    rVectors = []
    for i in range(len(vectors)):
        v = [0.0 for i in range(len(vectors[0][0]))]
        for j in range(len(vectors[0][0])):
            v[j] = vectors[i][2][j] - vectors[i][0][j] + vectors[i][1][j]
        rVectors.append(v)

    results = [['', 0.0] for i in range(len(analogies))]
    with open(word2vec_model, 'r') as fvec:
        count = 0
        fvec.next() # skip the first descriptive line
        for line in fvec:
            count += 1
            if count%100 == 0: print count
            if count > 30000: break

            line.strip('\n')
            list = line.split()
            word, vec = list[0], list[1:]
            vec = [float(vec[i]) for i in range(len(vec))]

            for i in range(len(vectors)):
                cosine = 1 - spatial.distance.cosine(rVectors[i], vec)
                if cosine > results[i][1]:
                    results[i][1] = cosine
                    results[i][0] = word
    return results

rsesults = addition()
POS = 0
for i in range(len(analogies)):
    if results[i][0] == analogies[i][3]:
        POS += 1
print "Accuracy of addition method: %s" % (float(POS)/float(len(analogies)) * 100)

