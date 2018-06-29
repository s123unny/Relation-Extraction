import gzip
import pickle as pkl
import numpy as np
import argparse
import nltk
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import treebank

def get_arg():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--test", default="data/test.txt")
	parser.add_argument("--train", default="data/train.txt")
	parser.add_argument("--output", "-o", default="pkl/data.pkl.gz", help="data pkl output path")
	parser.add_argument("--bin", action="store_true", help="define embeddings as word2vec bin file")
	parser.add_argument("--embeddingsPkl", "-p", default="pkl/embeddings.pkl.gz", help="embeddings pkl output path")
	parser.add_argument("--embeddings_bin", default="/tmp2/b05902005/GoogleNews-vectors-negative300.bin", help="bin type embeddings file ppath")
	parser.add_argument("--embeddings", default="/tmp2/b05902090/deps.words", help="text type embeddings file path")
	args = parser.parse_args()
	return args

#Mapping of the labels to integers
labelsMapping = {'Other':0, 
				 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
				 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
				 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
				 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
				 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
				 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
				 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
				 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
				 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

distanceMapping = {}
minDistance = -30
maxDistance = 30
for dis in range(minDistance,maxDistance+1):
	distanceMapping[dis] = len(distanceMapping)
#print (distanceMapping)
maxSentenceLen = [0,0]
words = {}
word2Idx = {}
pos2Idx = {}

def createMatices(file, maxSentenceLen):
	labels_map = []
	tokenIdsMatrix = []
	positionMatrix1 = []
	positionMatrix2 = []
	Pos_tagging = []
		
	for line in open(file, "r"):
		splits = line.strip().split('\t')
		
		label = splits[0]
		pos1 = int(splits[1])
		pos2 = int(splits[2])
		sentence = splits[3]
		tokens = sentence.split()
		pos_tagger = np.zeros(maxSentenceLen)
		tag = nltk.pos_tag(tokens)
		for i in range(len(tag)):
			if tag[i][1] not in pos2Idx:
				pos2Idx[ tag[i][1] ] = len(pos2Idx) + 1
			pos_tagger[i] = pos2Idx[ tag[i][1] ]

		tokenIds = np.zeros(maxSentenceLen)
		positionValues1 = np.zeros(maxSentenceLen)
		positionValues2 = np.zeros(maxSentenceLen)
		for idx in range(0, min(maxSentenceLen ,len(tokens))):
			tokenIds[idx] = getWordIdx(tokens[idx])
			
			distance1 = idx - pos1
			distance2 = idx - pos2
			
			if distance1 in distanceMapping:
				positionValues1[idx] = distanceMapping[distance1]
			elif distance1 <= minDistance:
				positionValues1[idx] = 0
			else:
				positionValues1[idx] = 60
				
			if distance2 in distanceMapping:
				positionValues2[idx] = distanceMapping[distance2]
			elif distance2 <= minDistance:
				positionValues2[idx] = 0
			else:
				positionValues2[idx] = 60
		
		labels_map.append(labelsMapping[label])
		tokenIdsMatrix.append(tokenIds)
		positionMatrix1.append(positionValues1)
		positionMatrix2.append(positionValues2)
		Pos_tagging.append(pos_tagger)

	return np.array(labels_map, dtype='int32'), np.array(tokenIdsMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'), np.array(Pos_tagging, dtype='int32')

def getWordIdx(token): 
	if token in word2Idx:
		return word2Idx[token]

	return word2Idx["UNKNOWN"]

def main():
	args = get_arg()

	files = [args.train, args.test]
	for fileIdx in range(2):
		file = files[fileIdx]
		for line in open(file):
			splits = line.strip().split('\t')
			sentence = splits[3]        
			tokens = sentence.split()
			maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
			for token in tokens:
				words[token] = True
	print ("Max sentence length:", maxSentenceLen)

	print ("embeddings...")
	embeddings = []  
	
	word2Idx["PADDING"] = 0
	vector = np.zeros(300)
	embeddings.append(vector)

	word2Idx["UNKNOWN"] = 1
	vector = np.random.uniform(-0.25, 0.25, 300)
	embeddings.append(vector)

	if args.bin:
		model = KeyedVectors.load_word2vec_format(args.embeddings_bin, binary=True)
		vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	else:
		vocab_list = []
		for line in open(args.embeddings):
			split = line.strip().split(" ")
			vocab_list.append([split[0], split[1:]])

	for i in range(len(vocab_list)):
		word = vocab_list[i][0].lower()
		
		if word in words:
			word2Idx[word] = len(word2Idx)
			embeddings.append(vocab_list[i][1])
		
	embeddings = np.array(embeddings)
	print ("Embeddings shape:", embeddings.shape)

	print ("dump into", args.embeddingsPkl)
	f = gzip.open(args.embeddingsPkl, 'wb')
	pkl.dump(embeddings, f, -1)
	f.close()

	print ("createMatices...")
	train_set = createMatices(args.train, max(maxSentenceLen))
	test_set = createMatices(args.test, max(maxSentenceLen))

	print ("dump into", args.output)
	f = gzip.open(args.output, 'wb')
	pkl.dump(train_set, f, -1)
	pkl.dump(test_set, f, -1)
	f.close()

if __name__ == "__main__":
	main()

