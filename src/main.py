from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Lambda, Activation, Concatenate, LeakyReLU, Add, Multiply, Conv1D, GlobalMaxPool1D
from bpemb import BPEmb

import numpy as np

WORD_EMBEDDINGS = BPEmb(lang="en", vs=10000)

UTTERANCE_EMBEDDING_DIM = 500  # Might as well be generous here - we're going to cluster the f**k out of them anyway
DICTIONARY_SIZE = 10  # 10 utterance types!

NUM_CONV_LAYERS = 5



def get_model():


    # QUANTIZING AUTO-ENCODER

    encoder_temperature = tf.Variable([1.0])

    # Input is a variable length sequence of integer tokens representing an utterance.
    # Keras hides the batch dimension from us - in reality the shape is (BATCH_SIZE, None)

    tokens = Input(shape=(None,))

    # Rather than fine-tuning the pre-trained word embeddings - which could destroy important information (esp. during early
    # exploratory optimization steps) - we keep them fixed and concatenate them with 100% learned embeddings

    pretrained_word_layer = Embedding(WORD_EMBEDDINGS.vocab_size, WORD_EMBEDDINGS.dim, weights=[WORD_EMBEDDINGS.vectors], trainable=False)

    learned_word_layer = Embedding(WORD_EMBEDDINGS.vocab_size, WORD_EMBEDDINGS.dim, weights=[np.zeros_like(WORD_EMBEDDINGS.vectors)], trainable=True)

    word_embeddings = Concatenate()([
        pretrained_word_layer(tokens),
        learned_word_layer(tokens),
    ])

    # Implement a wee multi-layer convolutional network

    layer_input = word_embeddings
    layer_outputs = []

    for layer in range(NUM_CONV_LAYERS):

        local_embeddings = Conv1D(WORD_EMBEDDINGS.dim, 3, padding="same")(layer_input)
        global_embedding = GlobalMaxPool1D()(local_embeddings)

        layer_outputs.append(global_embedding)

        layer_input = Concatenate()([
            local_embeddings,
            Lambda(lambda x: tf.expand_dims(x, axis=-2))(global_embedding)
        ])
        layer_input = Dense(WORD_EMBEDDINGS.dim)(layer_input)
        layer_input = LeakyReLU()(layer_input)

        layer_outputs.append(global_embedding)

    # Build an utterance embedding from the outputs

    utterance_embedding = Concatenate()(layer_outputs)
    utterance_embedding = Dense(UTTERANCE_EMBEDDING_DIM)(utterance_embedding)
    utterance_embedding = LeakyReLU()(utterance_embedding)
    utterance_embedding = Dense(UTTERANCE_EMBEDDING_DIM)(utterance_embedding)

    # Define the encoder, which takes an utterance embedding and outputs an (approximately) one-hot index into an embedding dictionary
    # Note: a high temperature decreases the input to the softmax, resulting in a more uniform output distribution and therefore
    # a "softer" clustering. Temperature should be gradually reduced during training to aid discovery of high quality clusters.

    encoder_input = Input(shape=(UTTERANCE_EMBEDDING_DIM,))

    encoder_output = Dense(UTTERANCE_EMBEDDING_DIM)(encoder_input)
    encoder_output = LeakyReLU()(encoder_output)
    encoder_output = Dense(DICTIONARY_SIZE)(encoder_output)
    encoder_output = Lambda(lambda x: x / (tf.reduce_sum(x, axis=-1) * encoder_temperature))(encoder_output)
    encoder_output = Activation("softmax")(encoder_output)

    encode = Model(encoder_input, encoder_output)

    # Define the decoder, which takes an (approximately) one-hot index, and "looks up" an embedding.

    decode = Dense(UTTERANCE_EMBEDDING_DIM, use_bias=False)  # That was easy, wasn't it?

    # Apply the encoder

    utterance_loadings = encode(utterance_embedding)

    # Apply the decoder

    decoded_utterance = decode(utterance_loadings)

    # Output the original embedding and its encoded-decoded version, for us to cast judgement upon.

    # Note, Keras has the constraint of one loss function per output, so we have to concatenate these into a single output,
    # and pick them apart inside out loss function :-(  An alternative would be to calculate the loss here inside our model,
    # pass that out, and use a simple pass-through (i.e. minimization) loss function, but its nice to have our model output
    # the embeddings for playing with :-)

    before_and_after_embeddings = Concatenate()([utterance_embedding, decoded_utterance])

    # Wrap the whole thing up into a model

    master_model = Model(tokens, [before_and_after_embeddings, utterance_loadings])

    return master_model, encoder_temperature


# LOSS FUNCTION

def batch_correlation_loss(y_true, y_pred):

        # We ignore y_true:
        # Keras requires that we pass in a "ground truth" by convention, but in our case all the information for our
        # loss is coming from inside our model

        original_embedding = y_pred[:, 0]
        reconstructed_embedding = y_pred[:, 1]

        magnitude = tf.sqrt(
            tf.reduce_sum(tf.square(original_embedding), axis=0) *
            tf.reduce_sum(tf.square(reconstructed_embedding), axis=0)
        )

        centered_original = original_embedding - tf.reduce_mean(original_embedding, axis=0)
        centered_reconstructed = reconstructed_embedding - tf.reduce_mean(reconstructed_embedding, axis=0)

        covar = tf.reduce_sum(centered_original * centered_reconstructed, axis=0)
        correl = covar / magnitude
        loss = - tf.reduce_mean(correl, axis=-1)

        return loss


# LET'S TRY IT OUT

model, temperature = get_model()
model.compile("adam", loss=[batch_correlation_loss, None])
