import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Download data bahasa (Wajib sekali saja)
nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

def tokenize(sentence):
    """Memecah kalimat menjadi array kata-kata"""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Mengambil akar kata (contoh: 'makan' -> 'makan')"""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Mengubah kalimat menjadi array angka 0 dan 1.
    Contoh:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1   ,  0 ,   1  ,   0  ,    0   ,    0  ]
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag