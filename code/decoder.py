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
        self.image_embedding1 = tf.keras.layers.Dense(
            hidden_size, activation='leaky_relu')
        self.image_embedding2 = tf.keras.layers.Dense(
            hidden_size)
        self.image_embedding = tf.keras.Sequential(
            [self.image_embedding1, self.image_embedding2])

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        # Define decoder layer that handles language and image context:
        self.decoder = tf.keras.layers.GRU(hidden_size, return_sequences=True)

        # Define classification layer (LOGIT OUTPUT)
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, activation='leaky_relu')  # make even bigger?
        self.dense2 = tf.keras.layers.Dense(vocab_size)

        self.classifier = tf.keras.Sequential([self.dense1, self.dense2])

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        imgs = self.image_embedding(encoded_images)
        wrds = self.embedding(captions)
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
        self.image_embedding1 = tf.keras.layers.Dense(
            hidden_size, activation='leaky_relu')
        self.image_embedding2 = tf.keras.layers.Dense(
            hidden_size)
        self.image_embedding = tf.keras.Sequential(
            [self.image_embedding1, self.image_embedding2])

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(
            vocab_size, hidden_size, window_size)  # need parens?

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(hidden_size)  # need parens?

        # Define classification layer (logits)
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, activation='leaky_relu')
        self.dense2 = tf.keras.layers.Dense(vocab_size)  # softmax? have probs?
        self.classifier = tf.keras.Sequential([self.dense1, self.dense2])

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        imgs = self.image_embedding(tf.expand_dims(encoded_images, 1))
        pos = self.encoding(captions)
        logits = self.decoder(pos, imgs)  # shape error coming from here? x
        probs = self.classifier(logits)
        return probs

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
        }
        return config
