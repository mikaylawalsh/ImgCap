import math
import ssl
import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys = K.get_shape()[1]  # window size of keys

        # Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        # This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal.
        # Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(
            np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(
            mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute attention weights using queries and key matrices
        #       - if use_mask==True, then make sure to add the attention mask before softmax
        # 2) return the attention matrix

        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matrix multiplied with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings. Return attention score as per lecture slides.

        score = (tf.matmul(Q, K, transpose_b=True)) / \
            tf.math.sqrt(tf.cast(window_size_keys, tf.float32))
        if self.use_mask:
            score = score + atten_mask
        return tf.nn.softmax(score)


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector
        self.K = tf.Variable(tf.random.normal(
            [input_size, output_size], stddev=0.01))
        self.Q = tf.Variable(tf.random.normal(
            [input_size, output_size], stddev=0.01))
        self.V = tf.Variable(tf.random.normal(
            [input_size, output_size], stddev=0.01))

        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrix products to turn inputs into keys, values, and queries.
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.

        K = tf.tensordot(inputs_for_keys, self.K, 1)
        V = tf.tensordot(inputs_for_values, self.V, 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, 1)

        mat = self.attn_mtx((K, Q))
        ret = mat@V

        return ret


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        # TODO: Add 3 heads as appropriate and any other necessary components

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        TODO: FOR CS2470 STUDENTS:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        return None


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, MultiHeadedAttention=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) For 2470 students, use multiheaded attention

        self.dense1 = tf.keras.layers.Dense(
            emb_sz, activation='leaky_relu')  # make even bigger?
        self.dense2 = tf.keras.layers.Dense(emb_sz)
        self.ff_layer = tf.keras.Sequential([self.dense1, self.dense2])

        self.self_atten = AttentionHead(
            emb_sz, emb_sz, True)
        self.self_context_atten = AttentionHead(
            emb_sz, emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        att = self.self_atten(inputs, inputs, inputs)
        res = att + inputs
        norm = self.layer_norm(res)
        att2 = self.self_context_atten(context_sequence, context_sequence, att)
        res2 = att2 + norm
        norm2 = self.layer_norm(res2)
        ff = self.ff_layer(norm2)
        res3 = ff + norm2
        norm3 = self.layer_norm(res3)
        ret = tf.nn.relu(norm3)

        return ret


def positional_encoding(length, depth):
    # REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    # TODO: Can remove signature
    depth = depth/2
    # Generate a range of positions and depths
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    # Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    # This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        # TODO: Implement Component

        # Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

        # Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies.
        # HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        # TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.

        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding
        return x
