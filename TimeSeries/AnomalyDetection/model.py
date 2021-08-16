import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GLU(layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate=None, use_time_distributed=True, activation=None):
        # Gated Linear Unit (GLU)
        super().__init__()
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = layers.Dropout(dropout_rate)
        
        if use_time_distributed:
            self.activation_layer = layers.TimeDistributed(layers.Dense(hidden_layer_size, activation=activation))
            self.gated_layer = layers.TimeDistributed(layers.Dense(hidden_layer_size, activation='sigmoid'))
        
        else:
            self.activation_layer = layers.Dense(hidden_layer_size, activation=activation)
            self.gated_layer = layers.Dense(hidden_layer_size, activation='sigmoid')

        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        if self.dropout_rate is not None:
            inputs = self.dropout(inputs)
        x = self.activation_layer(inputs)
        gate = self.gated_layer(x)
        return self.multiply([x, gate])

class GRN(layers.Layer):
    def __init__(self, features, hidden_size=64, drop=0.1, activation=None):
        # Gated Residual Network (GRN)
        super().__init__()
        self.dense1 = layers.Dense(hidden_size)
        self.elu = layers.ELU()
        self.dense2 = layers.Dense(features)
        self.dropout = layers.Dropout(drop)
        self.residual_add = layers.Add()
        self.gate = GLU(features, drop, activation=activation)
        self.bn = layers.LayerNormalization()
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.elu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.residual_add([x, inputs])
        x = self.bn(x)
        return x

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs



class temporal_head(layers.Layer):
    def __init__(self, embedding_dim, num_heads=8, hidden_embedding=256, hidden_size=256):
        super().__init__()
        self.lstm = layers.LSTM(hidden_embedding, return_sequences=True)
        self.grn = GRN(hidden_embedding, hidden_size=hidden_size)
        self.attn = MultiHeadAttention(embedding_dim, num_heads)
    
    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.grn(x)
        x = self.attn(x)
        return x

class feature_head(layers.Layer):
    def __init__(self, embedding_dim, temporal_dims, num_heads=8, hidden_embedding=256):
        super().__init__()
        self.lstm = layers.LSTM(hidden_embedding, return_sequences=False)
        self.repeat = layers.RepeatVector(temporal_dims)
        self.attn = MultiHeadAttention(embedding_dim, num_heads=num_heads)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.repeat(x)
        x = self.attn(x)
        return x

class Two_Step_Encoder(layers.Layer):
    def __init__(self, embedding_dim, temporal_dim, num_heads=8, hidden_embedding=256, hidden_size=256):
        super().__init__()
        self.temp_head = temporal_head(embedding_dim, num_heads, hidden_embedding, hidden_size)
        self.features = feature_head(embedding_dim, temporal_dim, num_heads, hidden_embedding)
        self.add = layers.Add()
        self.mlp = layers.Dense(embedding_dim)
    
    def call(self, inputs):
        x1 = self.temp_head(inputs)
        x2 = self.features(inputs)
        x = self.add([x1, x2])
        x = self.mlp(x)
        return x

class Decoder(layers.Layer):
    def __init__(self, features, hidden_dims=256):
        super().__init__()
        self.lstm = layers.LSTM(hidden_dims, return_sequences=True)
        self.dense1 = layers.Dense(hidden_dims)
        self.dense2 = layers.Dense(features)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
