import gzip
import pickle as pkl
import numpy as np
import argparse
import nltk
import re

def get_arg():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--test", default="data/TEST_FILE.txt")
	parser.add_argument("--train", default="data/TRAIN_FILE.txt")
	parser.add_argument("--answer", default="answer_key.txt")
	parser.add_argument("--output", default="data/")
	args = parser.parse_args()
	return args

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\.", " .", string)
	string = re.sub(r"\(", " ( ", string)
	string = re.sub(r"\)", " ) ", string)
	string = re.sub(r"\?", " ? ", string)
	return string.lower()

#train_set
def preprocess_train(train_file, outputfile):
	fOut = open(outputfile, "w")
	lines = [line.strip() for line in open(train_file)]
	for idx in range(0, len(lines), 4):
		sentence = lines[idx].strip().split("\t")[1][1:-1]
		label = lines[idx+1]

		sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " ")
		sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " ")
		tokens = nltk.word_tokenize(sentence)

		e1 = tokens.index("_e1_")
		del tokens[e1]
		
		e2 = tokens.index("_e2_")
		del tokens[e2]

		sentence = " ".join(tokens)
		sentence = clean_str(sentence)

		fOut.write("\t".join([label, str(e1), str(e2), sentence]))
		fOut.write("\n")
	fOut.close()

#test_set
def preprocess_test(test_file, key_file, outputfile):
	fOut = open(outputfile, "w")
	labels = []
	for line in open(key_file):
		label = line.strip().split("\t")[1]
		labels.append(label)

	with open(test_file) as f:
		for idx, line in enumerate(f):
			sentence = line.strip().split("\t")[1][1:-1]

			sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " ")
			sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " ")
			tokens = nltk.word_tokenize(sentence)

			e1 = tokens.index("_e1_")
			del tokens[e1]
			
			e2 = tokens.index("_e2_")
			del tokens[e2]

			sentence = " ".join(tokens)
			sentence = clean_str(sentence)
			
			fOut.write("\t".join([labels[idx], str(e1), str(e2), sentence]))
			fOut.write("\n")
	fOut.close()

def main():
	args = get_arg()
	preprocess_train(args.train, args.output+"train.txt")
	preprocess_test(args.test, args.answer, args.output+"test.txt")

if __name__ == "__main__":
	main()