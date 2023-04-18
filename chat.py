from keras.models import load_model
from prepare_data import data
from preprocess_data import tokenizer, maxlen
import numpy as np

model = load_model('funchatbot.h5')


while True:
    inputs = input('You: ')
    if inputs == 'break':
        break
    
    x = tokenizer.texts_to_tensor([inputs], maxlen)
    y = model(x)
    y_max = np.max(y)
    
    predictions = np.argmax(y)
    
    if y_max > 0.6:
        print(y_max)
        print(data[predictions][1] + '\n')
    else:
        print(y_max)
        print("Sorry I do not understand")