# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model to detect sarcasm in text.
import json
import numpy
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters.
token_size = 3000
embedding_dim = 16
text_length = 120
epochs = 10
training_size = 20000
batchSize = 128

# Get data from sarcasm.json file.
with open("./sarcasm.json", 'r') as file:
    data = json.load(file)

# Save as sentences and lables.
sentences = []
labels = []
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# Prepare training data.
training_sentences = sentences[0:training_size]
training_labels = numpy.array(labels[0:training_size])
# Prepare validation data.
validation_sentences = sentences[training_size:]
validation_labels = numpy.array(labels[training_size:])

# Create tokenizer.
tokenizer = Tokenizer(num_words=token_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Get text from word tokens.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '') for i in text])

# Create training sequences.
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_data = numpy.array(pad_sequences(
    training_sequences, maxlen=text_length, padding="post", truncating="post"))


# Create validation sequences.
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_data = numpy.array(pad_sequences(
    validation_sequences, maxlen=text_length, padding="post", truncating="post"))

# Create model with 1 output unit for classification.
model = keras.Sequential([
    keras.layers.Embedding(token_size, embedding_dim,
                           input_length=text_length),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Set loss function and optimizer.
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(training_data, training_labels, epochs=epochs, callbacks=[
          checkAccuracy], batch_size=batchSize, validation_data=(validation_data, validation_labels), verbose=1)


# Predict on a random validation text.
index = 7
text = validation_data[index]
prediction = model.predict(text.reshape(1, 120, 1))

print("Prediciton : ", prediction[0][0] )
print("Label : " , validation_labels[index])
print("Text : ", decode_sentence(validation_data[index]))