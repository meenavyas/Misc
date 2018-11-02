
# coding: utf-8

# ### This notebook reads openssl .c source code and generates new source code
# #### References
# ##### https://www.tensorflow.org/tutorials/sequences/recurrent
# ##### https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# 
# ### Prerequisites install tensorflow and keras
# #### pip install tensorflow
# #### pip install keras

# In[1]:


import numpy
import os
import sys
import time

from keras.models import Sequential 
from keras.layers import Activation 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


# ### Tune these parameters

# In[2]:


SEQ_LENGTH = 100
EPOCHS = 10
BATCH_SIZE = 128


# ### Preprocessing 
# #### read all files and concatenate into a single file train.txt
# #### Add tags "<start"> and "< eof >"

# In[3]:


PWD=os.getcwd()
print("PWD=" + str(PWD))
train_filename = PWD+"./train.txt"

readSize = 0

if (os.path.isfile(train_filename)):
    os.remove(train_filename)
    
# read every 3rd file otherwise getting out of memory errors
i = 0
with open(train_filename, "w") as a:
    for path, subdirs, files in os.walk(PWD+"/openssl-master"):
        for filename in files:
            if (filename.endswith(".c") and (i%3 == 0)):
                #print("Reading "+ filename)
                f = os.path.join(path, filename)
                readSize += os.path.getsize(f)
                currfile = open(f).read()
                a.write("<start>")
                a.write(currfile)
                a.write("<eof>")
                i=i+1
                
print("readSize="+str(readSize))


# ### Open "train.txt" convert into lower case and sort them 
# 

# In[4]:


raw_text = open(train_filename).read().lower()
INPUT_FILE_LEN = len(raw_text)

chars = sorted(list(set(raw_text)))
VOCAB_LENGTH = len(chars)

print ("Length of file: "+ str(INPUT_FILE_LEN))
print ("Total Vocab length (unique chars in input) : "+ str(VOCAB_LENGTH))


# ### create mapping of unique chars to integers, and a reverse mapping
# 

# In[5]:


char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print(char_to_int)
print(int_to_char)


# In[6]:


dataX = []
dataY = []
for i in range(0, INPUT_FILE_LEN - SEQ_LENGTH, 1):
    seq_in = raw_text[i:i + SEQ_LENGTH]
    seq_out = raw_text[i + SEQ_LENGTH]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

samples = len(dataX)
print( "Total samples: "+ str(samples))

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (samples, SEQ_LENGTH, 1))

# normalize
X = X / float(VOCAB_LENGTH)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)
print("X.shape=" + str(X.shape))
print("y.shape=" + str(y.shape))


# ### create model with 2 LSTM Layers with dropout 0.2, 1 Dense layer with softmax
# 

# In[7]:


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2)) # 0.5
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam') 


# ### Visualize the model

# In[8]:


print(model.summary())


# ### Fit the model

# In[9]:


histroy = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)


# ### Plot Model Loss 

# In[10]:


import matplotlib.pyplot as plt 
# list all data in history
print(histroy.history.keys())

# summarize history for loss
plt.plot(histroy.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# ### Generate the code

# In[19]:


start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("input code starts with: [", ''.join([int_to_char[value] for value in pattern]), "]")
# generate characters
for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(VOCAB_LENGTH)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\n#####.")

