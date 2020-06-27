# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:45:37 2020

@author: ACER
"""

import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

with open(r"C:\Users\ACER\Desktop\AI\NLP\train_qa.txt",'rb') as f:
    train_data = pickle.load(f)
    
with open(r"C:\Users\ACER\Desktop\AI\NLP\test_qa.txt",'rb') as f:
    test_data = pickle.load(f)
    
#type of training and test data = list
# Train data and test data, each entry is a tuple with 2 lists and a string. 
#The 2 lists are the story, the question and the string is the answer

vocab = set() # For vocab. Both are lists and so appending

all_data = test_data + train_data
for story, question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


vocab.add('no')
vocab.add('yes')

vocab_len = len(vocab) + 1 #for pad sequences
max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])

vocab_size = len(vocab) + 1
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)# tokenizer also lower cases
tokenizer.word_index

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)# a list 
    
train_story_seq = tokenizer.texts_to_sequences(train_story_text)

def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len):
    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []
    
    
    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        # Index 0 is reserved so we're going to use + 1 for padding
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
    # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

tokenizer.word_index['yes']

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM

input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

# Input gets embedded to a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))

# This encoder will output:
# (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_question_len))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)  
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer) #(batch_size,vocab_size) where everything will be 0s except for yes/no indeices

answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))

model.save('My_chatbot.h5')


pred_results = model.predict(([inputs_test, queries_test]))


#testing the model
test_data[0][0] #story
test_data[0][1] #question
test_data[0][2] # answer here it is no

val_max = np.argmax(pred_results[0]) #since pred_results[0] is an array of length of vocab, with probs in the indexes of words showing confidence
#above statement returns index of word in vocab with max prob
for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])


#Making our own story. Note: We have to format the story in the same way with the same vocabulary
my_story = "Daniel took the apple to the office . Mary discarded the milk in the kitchen ."
my_question = "Is milk in the kitchen ?"
mydata = [(my_story.split(),my_question.split(),'yes')]
my_story,my_ques,my_ans = vectorize_stories(mydata)

pred_results = model.predict(([ my_story, my_ques]))
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
    








