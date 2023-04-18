import numpy as np
import re
from keras.utils import to_categorical

from prepare_data import questions, labels

np.random.seed(81)

class __Tokenizer__:
    
    def __init__(self):
        self.words = None
        self.words_map = None
    
    def fit_texts(self, text_list):
        all_words = []
        for texts in text_list:
            clean_text = re.sub(r'[^\w\s]', '', texts.lower())
            words = list(set(str(clean_text).split()))
            all_words.extend(words)
        self.words = sorted(set(all_words))
        
        self.words_map = {word: i + 1 for i, word in enumerate(self.words)}
        self.words_reverse = {i + 1: word for i, word in enumerate(self.words)}
        
        self.words_map.update({"[UNK]": 0})
        self.words_reverse.update({0: "[UNK]"})
        
        return self
                
    def get_sequences(self, inputs):
        self.sequences = []
        for inp in inputs:
            clean_inp = re.sub(r'[^\w\s]', '', inp.lower())
            words = str(clean_inp).split()
            sequence = [self.words_map[word] if word in self.words else np.inf for word in words]
            self.sequences.append(sequence)
        return self.sequences
    
    def stuffing(self, sequences, maxlen: int, padding='pre', cut='pre'):
        arr = np.zeros((len(sequences), maxlen))
        
        for i, sequence in enumerate(sequences):
            
            if len(sequence) > maxlen:
                if cut == 'pre':
                    sequence = sequence[(len(sequence) - maxlen):]
                elif cut == 'post':
                    sequence = sequence[:maxlen]
        
            if padding == 'pre':
                arr[i][:len(sequence)] = np.array(sequence)
            elif padding == 'post':
                arr[i][(maxlen - len(sequence)):] = np.array(sequence)
        return arr
    
    def decrypt(self, sequences):
        self.results = []
        for sequence in sequences:
            result = [self.words_reverse[s] for s in sequence]
            self.results.append(result)
        return self.results
    
    def texts_to_tensor(self, text_list, maxlen, padding='pre', cut='pre'):
        sequences = self.get_sequences(text_list)
        s_tensor = self.stuffing(sequences, maxlen, padding, cut)
        return s_tensor
   

tokenizer = __Tokenizer__().fit_texts(questions)

maxlen = 15
input_dim = len(tokenizer.words)

X = tokenizer.texts_to_tensor(questions, maxlen)
y = to_categorical(labels, len(questions))
indices = np.random.permutation(len(X))

X_train = X[indices].astype(np.float32)
y_train = y[indices].astype(np.float32)