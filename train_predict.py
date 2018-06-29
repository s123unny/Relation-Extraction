import numpy as np
import pickle as pkl
import gzip
import keras
import argparse
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

def get_arg():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--data", "-d", default="pkl/data.pkl.gz", help="data pkl path")
	parser.add_argument("--embeddings", "-e", default="pkl/embeddings.pkl.gz", help="embeddings pkl path")
	parser.add_argument("--predict", "-p", default="result/predict.txt", help="save predict path")
	parser.add_argument("--model", "-m", default="result/CNNmodel.h5", help="save model path")
	parser.add_argument("--load", "-l", default=None, help="load model and do predict only")
	args = parser.parse_args()
	return args

#Mapping of the integers to labels
labelsMapping = {0: 'Other',
                 1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                 3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                 5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                 7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                 9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                 11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                 13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                 15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                 17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}
batch_size = 64
nb_filter = 100
kernel_size = 3
hidden_dims = 100
nb_epoch = 25
position_dims = 50

args = get_arg()
f = gzip.open(args.data, 'rb')
yTrain, tokenIdTrain, positionTrain1, positionTrain2, POStagTrain = pkl.load(f)
yTest, tokenIdTest, positionTest1, positionTest2, POStagTest = pkl.load(f)
f.close()
if args.load != None:
	model = load_model(args.load)
else:
	max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1
	max_postag = max(np.max(POStagTrain), np.max(POStagTest))+1
	max_sentence_len = positionTrain1.shape[1]

	n_out = max(yTrain)+1
	train_y_cat = np_utils.to_categorical(yTrain, n_out)

	f = gzip.open(args.embeddings, 'rb')
	embeddings = pkl.load(f)
	f.close()

	print("yTrain: ", yTrain.shape)
	print ("tokenIdTrain:", tokenIdTrain.shape)
	print ("positionTrain1:", positionTrain1.shape)
	print ("positionTrain2:", positionTrain1.shape)
	print ("POStagTrain:", POStagTrain.shape)
	print ("embeddings:", embeddings.shape)

	distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
	distance1 = Embedding(max_position, position_dims)(distance1_input)

	distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
	distance2 = Embedding(max_position, position_dims)(distance2_input)

	POStag_input = Input(shape=(max_sentence_len,), dtype='int32', name='POStag_input')
	POStag = Embedding(max_postag, position_dims)(POStag_input)

	words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
	words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)
	#words = Dense(position_dims)(words)

	output = concatenate([words, distance1, distance2])

	output = Convolution1D(filters=nb_filter,
	                        kernel_size=kernel_size,
	                        strides=1,
	                        padding="same",
	                        activation='relu')(output)
	# we use standard max over time pooling
	output = GlobalMaxPooling1D()(output)

	output = Dropout(0.25)(output)
	output = Dense(n_out, activation='softmax')(output)

	earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(
		filepath=args.model, verbose=1, save_best_only=True, monitor='val_loss', mode='auto')

	model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	model.summary()
	print ("Start training")

	max_acc = 0      
	model.fit([tokenIdTrain, positionTrain1, positionTrain2], train_y_cat, 
		batch_size=batch_size, verbose=True, epochs=nb_epoch) 
		#batch_size=batch_size, verbose=True, epochs=nb_epoch, validation_split=0.05, callbacks=[checkpoint, earlystopping])   

print ("Start predicting")
pred_test = model.predict([tokenIdTest, positionTest1, positionTest2], verbose=False)

dctLabels = np.sum(pred_test)
totalDCTLabels = np.sum(yTest)

class_test = [np.argmax(y, axis=None, out=None) for y in pred_test]
class_test = np.array(class_test)
acc =  np.sum(class_test == yTest) / float(len(yTest))

print ("Write predict into", args.predict)
print ("class_test:", class_test.shape)
outputfile = open(args.predict, "w")
for i in range(len(class_test)):
	outputfile.write("{}\t{}\n".format(i+8001, labelsMapping[class_test[i]]))


if args.load == None:
 	print ("Save model to", args.model)
 	model.save(args.model)
