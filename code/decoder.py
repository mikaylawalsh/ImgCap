import tensorflow as tf

try:
    from transformer import TransformerBlock, PositionalEncoding
except Exception as e:
    print(
        f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################


class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer to embed image features into a vector
        # with the models hidden size ??
        self.image_embedding = tf.keras.layers.Dense(
            hidden_size, activation='leaky_relu')

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        # Define decoder layer that handles language and image context:
        self.decoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

        # Define classification layer (LOGIT OUTPUT)
        self.dense1 = tf.keras.layers.Dense(
            500, activation='leaky_relu')  # make even bigger?
        self.dense2 = tf.keras.layers.Dense(vocab_size)

        self.seq = tf.keras.Sequential([self.dense1, self.dense2])

        self.classifier = self.seq

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        print(encoded_images.shape)
        print(captions.shape)
        imgs = self.image_embedding(encoded_images)
        wrds = self.embedding(captions)
        # how to pass 2 arguments
        logits = self.decoder(wrds, initial_state=imgs)
        logits = self.classifier(logits)

        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector
        self.image_embedding = None

        # Define positional encoding to embed and offset layer for language:
        self.encoding = None

        # Define transformer decoder layer:
        self.decoder = None

        # Define classification layer (logits)
        self.classifier = None

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        probs = None
        return probs
