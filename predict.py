import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Import model
with open('models/model.pkl', 'rb') as mod:
    model = pickle.load(mod)
# Import Tokenizer
with open('models/tokenizer.pkl', 'rb') as tok:
    tokenizer = pickle.load(tok)

EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"

def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the correct padding, truncating and maxlen
    pad_trunc_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    
    return pad_trunc_sequences

def predict_comment(tokenizer, comment, cat):
    # Preprocess comment sesuai dengan tokenizer yang digunakan
    comment_seq = tokenizer.texts_to_sequences([comment])
    comment_pad = pad_sequences(comment_seq, maxlen=MAXLEN, padding=PADDING)
    
    # Lakukan prediksi
    pred_prob = model.predict(comment_pad)
    
    # Ambil indeks kategori dengan probabilitas tertinggi
    pred_labels = np.argmax(pred_prob)
    
    # Komentar ini termasuk dalam kategori apa?
    predicted_category = cat[pred_labels]
    
    return pred_prob, predicted_category, pred_labels
