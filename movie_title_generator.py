import tflearn
from tflearn.data_utils import *

#printing type of object/value passed
def showtype(someobject):
    print(type(someobject))
    return

#print random text
def printrandomtext(trained_model, seed):
    randomtemp=random.uniform(0.5, 1.9)
    randomtxt=trained_model.generate(30, randomtemp, seq_seed=seed)
    print("__________")
    print("novelty {0}".format(randomtemp))
    print("seed")
    print(seed)
    print("random text")
    print(randomtxt)

#settings
sequence_max_length = 20
path = "movies.txt"

#load movies titles as one string
existing_movies_titles_as_string = open(path).read()

#preprocess data
X, Y, character_indexes = \
    string_to_semi_redundant_sequences(existing_movies_titles_as_string, seq_maxlen=sequence_max_length, redun_step=3)

#length of character dictionary
how_many_unique_characters=len(character_indexes)

#define model
net = tflearn.input_data(shape=[None, sequence_max_length, how_many_unique_characters])
net = tflearn.lstm(net, 128, return_seq=True)
net = tflearn.dropout(net, 0.5)
net = tflearn.lstm(net, 128)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, how_many_unique_characters, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',
                         learning_rate=0.001)

model = tflearn.SequenceGenerator(net, dictionary=character_indexes,
                                  seq_maxlen=sequence_max_length,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_movies',
                                  tensorboard_verbose=3) #full log

#train model 40 times and show results every step
for i in range(40):
    model.fit(X, Y, validation_set=0.2, batch_size=128,
              n_epoch=1, run_id='movies')
#generate some text based on what learned

    seed = random_sequence_from_string(existing_movies_titles_as_string, sequence_max_length)
    print("epoch: {0}".format(i))
    for j in range(10):
        printrandomtext(model, seed)