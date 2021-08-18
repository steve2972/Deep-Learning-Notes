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
        self.dense3 = layers.Dense(features)
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
        self.lstm2 = layers.LSTM(embedding_dim, return_sequences=True)
        self.grn = GRN(embedding_dim, hidden_size=hidden_size)
        #self.attn = MultiHeadAttention(embedding_dim, num_heads)
    
    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.lstm2(x)
        x = self.grn(x)
        #x = self.attn(x)
        return x

class feature_head(layers.Layer):
    def __init__(self, embedding_dim, temporal_dims, num_heads=8, hidden_embedding=256):
        super().__init__()
        self.lstm = layers.LSTM(hidden_embedding, return_sequences=False)
        self.repeat = layers.RepeatVector(temporal_dims)
        self.dense= layers.Dense(hidden_embedding, activation='elu')
        self.dense1 = layers.Dense(hidden_embedding, activation='elu')
        self.attn = MultiHeadAttention(embedding_dim, num_heads=num_heads)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        x = self.dense1(x)
        x = self.repeat(x)
        x = self.attn(x)
        return x

class TwoStepEncoder(layers.Layer):
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
    
class TwoStepDecoder(tf.keras.Model):
    def __init__(self, num_features, window_size, hidden_size=64):
        super().__init__()        
        self.temp_head = temporal_head(hidden_size)
        self.features = feature_head(hidden_size, window_size)
        self.add = layers.Add()
        self.dense = layers.Dense(num_features, activation='relu')
        self.mlp = layers.Dense(num_features, activation='sigmoid')
        
                                       
                                       
    def call(self, inputs):
        x1 = self.temp_head(inputs)
        x2 = self.features(inputs)
        x = self.add([x1, x2])
        x = self.mlp(self.dense(x))

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=1, strides=1, padding='same', dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel = kernel_size
        self.strides = strides
        self.pad = padding
        self.conv = layers.Conv1D(self.filters, kernel_size=self.kernel, strides=self.strides, padding=self.pad, dilation_rate=dilation_rate)
        self.elu = layers.ELU()
                      
        
    def call(self, inputs):
        x = self.conv(inputs)        
        x = self.elu(x)
        
        return x

class ConvHead(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=1, pool_size=1, dilation_rate=1):
        super(ConvHead, self).__init__()
        self.filters = filters        
        self.conv1 = ConvBlock(filters=self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = ConvBlock(filters=self.filters, kernel_size=1)
        self.conv3 = ConvBlock(filters=self.filters, kernel_size=1)
        self.maxpool = layers.MaxPool1D(pool_size=pool_size)  
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        
        return x
    
class DenseGroup(tf.keras.layers.Layer):
    def __init__(self, units=256, activation=None):
        super(DenseGroup, self).__init__()           
        self.dense1 = layers.Dense(units, activation=activation)
        self.dense2 = layers.Dense(units, activation=activation)
        self.dense3 = layers.Dense(units, activation=activation)
          
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)        
        
        return x
    
    
class MultiheadEncoder(tf.keras.Model):
    def __init__(self, num_features, window_size, use_attention=False):
        super(MultiheadEncoder, self).__init__()
        self.use_attention = use_attention
        self.num_features=num_features
        self.window_size=window_size

        self.recurrent1 = layers.LSTM(64, return_sequences=True)              
        self.denseG1 = [DenseGroup(units=64, activation='elu') for _ in range(self.window_size)]        
        
        self.permute1 = layers.Permute((2,1))
        self.permute2 = layers.Permute((2,1))        
        
        self.dense1 = layers.Dense(512, activation='elu')
        self.dense2 = layers.Dense(512, activation='elu')
        self.dense3 = layers.Dense(32, activation='elu') 
        
        if use_attention:
            self.attn1 = MultiHeadAttention(32, 8)
            self.attn2 = MultiHeadAttention(64, 8)        
        #'''
        self.recurrent10 = layers.LSTM(128, return_sequences=True)        
        self.recurrent11 = layers.LSTM(128, return_sequences=True)
        self.recurrent12 = layers.LSTM(64, return_sequences=True)
        #'''
        
        self.permute10 = layers.Permute((2,1))
        self.permute11 = layers.Permute((2,1))
        
        self.dense10 = layers.Dense(512, activation='elu')
        self.dense11 = layers.Dense(512, activation='elu')
        self.dense12 = layers.Dense(256, activation='elu')
        
        
        #self.grn = GRN(64)
        
    def call(self, inputs):
        x = self.recurrent1(inputs)    
        
        x1 = [self.denseG1[i](x[:,i]) for i in range(self.window_size)]
        x2 = [tf.expand_dims(x1[i], axis=1) for i in range(self.window_size)]
        x = tf.concat(x2, axis=1)
        
        x = self.permute1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        if self.use_attention:
            x = self.attn1(x)
        x = self.permute2(x)
        
        y = self.recurrent10(inputs)
        y = self.recurrent11(y)
        y = self.recurrent12(y) 
        #y = self.grn(inputs)
        
        y = self.permute10(y)
        y = self.dense10(y)
        y = self.dense11(y)
        y = self.dense12(y)
        y = self.permute11(y)
        
        x = tf.concat([x, y], axis=1)
        if self.use_attention:
            x = self.attn2(x)

        
        return x
                                       
class MultiheadDecoder(tf.keras.Model):
    def __init__(self, num_features, window_size):
        super(MultiheadDecoder, self).__init__()        
        
        self.num_features = num_features
        self.window_size = window_size
                
        self.permute1 = layers.Permute((2,1))
        self.permute2 = layers.Permute((2,1))
        self.dense1 = layers.Dense(512, activation='elu')
        self.dense2 = layers.Dense(512, activation='elu')
        self.dense3 = layers.Dense(self.window_size, activation='elu')
        
        self.recurrent1 = layers.LSTM(128, return_sequences=True)        
        self.recurrent2 = layers.LSTM(128, return_sequences=True)    
        self.recurrent3 = layers.LSTM(64, return_sequences=True)
        
        self.permute10 = layers.Permute((2,1))
        self.permute11 = layers.Permute((2,1))   
        
        self.dense10 = layers.Dense(512, activation='elu')
        self.dense11 = layers.Dense(512, activation='elu')
        self.dense12 = layers.Dense(self.window_size, activation='elu')
             
        self.dense21 = layers.Dense(128, activation='elu')
        self.dense22 = layers.Dense(128, activation='elu')
        self.dense23 = layers.Dense(self.num_features, activation='sigmoid')        
        
                                       
                                       
    def call(self, inputs):    
        
        x = inputs[:,:32,:]
        x = self.permute1(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        x = self.permute2(x)
       
        
        y = self.recurrent1(inputs[:,32:,:])        
        y = self.recurrent2(y)                
        y = self.recurrent3(y)                        
        
        y = self.permute10(y)
        y = self.dense10(y)
        y = self.dense11(y)
        y = self.dense12(y) 
        y = self.permute11(y)
        
        x = x + y
                
        x = self.dense21(x)
        x = self.dense22(x)        
        x = self.dense23(x) 
        
        return x
    
    
class Autoencoder(tf.keras.Model):
    def __init__(self, num_features, window_size, batch_size, use_attention=False):
        super(Autoencoder, self).__init__()
        self.encoder = MultiheadEncoder(num_features=num_features, window_size=window_size, use_attention=use_attention)
        self.decoder = MultiheadDecoder(num_features=num_features, window_size=window_size)
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Autoencoder_attn(tf.keras.Model):
    def __init__(self, num_features, window_size, batch_size):
        super(Autoencoder_attn, self).__init__()
        self.encoder = TwoStepEncoder(32, window_size, num_heads=8, hidden_embedding=256, hidden_size=256)
        self.decoder = tf.keras.Sequential([
            layers.LSTM(256, return_sequences=True),
            layers.TimeDistributed(layers.Dense(256)),
            layers.TimeDistributed(layers.Dense(256)),
            layers.Dense(num_features, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
