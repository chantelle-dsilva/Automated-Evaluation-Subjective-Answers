import gensim
from os import listdir
from os.path import isfile, join
from scipy import spatial

LabeledSentence = gensim.models.doc2vec.LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.doc_list = filename
    def __iter__(self):
    	for i in range(3):
	        for uid, line in enumerate(open(filename[i])):
	        	yield LabeledSentence(words=line.split(), tags=['SENT_%s_%s' % (i,uid)])
        		print (str(i) + " ___ " +str(uid))

docLabels = ['student responses 1.1.txt','student responses 2.7.txt','student responses 6.1.txt']
filename = []
for doc in docLabels:
  filename.append('E:/BEProject/short-answer-grader/Sample Train Data/' + doc)

sentence = LabeledLineSentence(filename)
model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(sentence)

for epoch in range(10):
    model.train(sentence,total_examples = model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(sentence,total_examples = model.corpus_count,epochs=model.iter)

model.save('doc2vec.model')

print (model.docvecs.most_similar("SENT_1_0"))
print (1 - spatial.distance.cosine(model["SENT_1_0"],model["SENT_1_3"]))