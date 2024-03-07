import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model

def hierarchical_attention_model(max_sentences, max_words, embedding_dim, class_num):
    # Define input layers for sentences and words.
    sentence_input = Input(shape=(max_words,), dtype='int32')
    word_input = Input(shape=(max_words,), dtype='int32')
    
    # Embed the word inputs and apply a bidirectional LSTM.
    word_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(word_input)
    word_lstm = Bidirectional(LSTM(units=embedding_dim, return_sequences=True))(word_embedding)
    # Compute attention weights for each word and apply them.
    word_attention = TimeDistributed(Dense(1, activation='tanh'))(word_lstm)
    word_attention = tf.squeeze(word_attention, axis=-1)
    word_attention = tf.nn.softmax(word_attention)
    
    # Repeat the process above for sentences.
    sentence_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(sentence_input)
    sentence_lstm = Bidirectional(LSTM(units=embedding_dim, return_sequences=True))(sentence_embedding)
    sentence_attention = TimeDistributed(Dense(1, activation='tanh'))(sentence_lstm)
    sentence_attention = tf.squeeze(sentence_attention, axis=-1)
    sentence_attention = tf.nn.softmax(sentence_attention)
    sentence_representation = tf.reduce_sum(sentence_lstm * tf.expand_dims(sentence_attention, axis=-1), axis=1)
    
    # Combine the sentence representations into a document-level output.
    document_output = Dense(units=class_num, activation='softmax')(sentence_representation)
    
    # Compile and return the model.
    model = Model(inputs=[sentence_input, word_input], outputs=document_output)
    return model

# Sample documents and labels for training.
documents = [
    "I love this movie. The acting was great and the story was captivating.",
    "The food at the restaurant was delicious. I highly recommend it.",
    "I didn't enjoy the concert. The music was too loud and the crowd was rowdy."
]
labels = [1, 1, 0]  

# Prepare text data for training.
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(documents)
word_sequences = tokenizer.texts_to_sequences(documents)

# Pad sequences to a consistent length.
max_words = max([len(seq) for seq in word_sequences])
word_sequences = pad_sequences(word_sequences, maxlen=max_words)

# Prepare sentence and label arrays.
max_sentences = len(documents)
sentence_sequences = np.array([np.arange(max_words)] * max_sentences)
labels = np.array(labels)
vocab_size = len(tokenizer.word_index) + 1

# Model parameters and compilation.
embedding_dim = 100
class_num = 2  
model = hierarchical_attention_model(max_sentences, max_words, embedding_dim, class_num)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model.
epochs = 10
batch_size = 2
model.fit([sentence_sequences, word_sequences], labels, epochs=epochs, batch_size=batch_size)

# Prepare and predict new documents.
new_documents = [
    "The book was incredible. It kept me hooked from start to finish.",
    "The service at the hotel was terrible. I wouldn't recommend staying there."
]
new_word_sequences = tokenizer.texts_to_sequences(new_documents)
new_word_sequences = pad_sequences(new_word_sequences, maxlen=max_words)
new_sentence_sequences = np.array([np.arange(max_words)] * len(new_documents))
predictions = model.predict([new_sentence_sequences, new_word_sequences])
predicted_labels = np.argmax(predictions, axis=1)

# Output predictions for new documents.
for i, doc in enumerate(new_documents):
    print(f"Document: {doc}")
    print(f"Predicted Label: {predicted_labels[i]}")


