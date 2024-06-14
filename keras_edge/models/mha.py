import keras
import numpy as np

DEBUG=False
class MultiHeadAttention(keras.layers.Layer):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """
    def __init__(
            self,
            n_head: int,
            n_feat: int,
            dropout_rate: float = 0.0,
            max_cache_len: int = 0,
            kernel_initializer=None,
            bias_initializer=None,
        ):
        """MultiHeadedAttention."""
        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head

        self.s_d_k = keras.ops.sqrt(keras.ops.cast(self.d_k, "float32"))
        self.h = n_head
        self.linear_q = keras.layers.Dense(
            n_feat,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        self.linear_k = keras.layers.Dense(
            n_feat,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        self.linear_v = keras.layers.Dense(
            n_feat,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        self.linear_out = keras.layers.Dense(
            n_feat,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.permutation=[0, 2, 1, 3]
        self._max_cache_len = max_cache_len

    def forward_qkv(
            self,
            query: keras.KerasTensor,
            key: keras.KerasTensor,
            value: keras.KerasTensor):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        # import numpy as np
        # np.random.seed(0)
        # a = np.random.randn(1,266, 176)
        # a = keras.Variable(a,dtype=tf.float32)
        # print("a")
        # print(a[0,:5,:5])
        # n_batch = key.shape[0]
        # b = self.linear_v(a)
        # print("b")
        # print(b[0,:4,:4])
        n_batch = keras.ops.shape(query)[0]
        shape=(n_batch, -1, self.h, self.d_k)
        q = keras.ops.reshape(self.linear_q(query), shape)
        k = keras.ops.reshape(self.linear_k(key),   shape)
        v = keras.ops.reshape(self.linear_v(value), shape)

        q = keras.ops.transpose(q, self.permutation)
        k = keras.ops.transpose(k, self.permutation)
        v = keras.ops.transpose(v, self.permutation)

        return q, k, v

    def forward_attention(
            self,
            value: keras.KerasTensor,
            scores:keras.KerasTensor,
            mask: keras.KerasTensor,
            training:bool=False):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model)
            weighted by the attention scores
        """

        n_batch = keras.ops.shape(value)[0]
        # if mask is not None:
        #     mask = tf.expand_dims(mask, 1)  # (batch, 1, time1, time2)
        #     scores = scores.masked_fill(mask, -10000.0)
        #     attn = keras.ops.softmax(scores, axis=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        # else:
        #     attn = keras.ops.softmax(scores, axis=-1)  # (batch, head, time1, time2)
        if mask is None:
            attn = keras.ops.softmax(scores, axis=-1)  # (batch, head, time1, time2)
        else:
            mask = keras.ops.expand_dims(mask, 1)
            # mask = tf.pad(
            #     mask,
            #     [[0,0],
            #      [0,0],
            #     [0,scores.shape[2]-mask.shape[2]],
            #     [0,scores.shape[2]-mask.shape[2]]])
            # mask = mask[:,:,:scores.shape[2],:scores.shape[2]]
            scores = keras.ops.where(mask > 0, scores, -10000)
            attn = keras.ops.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn, training=training)
        x = keras.ops.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = keras.ops.transpose(x, self.permutation) # (batch, time1, heads, d_k)
        x = keras.ops.reshape(x, (n_batch, -1, self.h * self.d_k))   # (batch, time1, d_model=heads*d_k)

        return self.linear_out(x)  # (batch, time1, d_model)

    def call(
            self,
            query: keras.KerasTensor,
            key: keras.KerasTensor,
            value: keras.KerasTensor,
            mask: keras.KerasTensor = 1,
            pos_emb:keras.KerasTensor =None,
            cache: keras.KerasTensor=None,
            training: bool =False):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor):
                transformed `value` (batch, time1, d_model)
                weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0,1,3,2))) / self.s_d_k
        out = self.forward_attention(v, scores, mask, training=training)
        return out

class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float =0.0,
        pos_bias_u=None,
        pos_bias_v=None,
        max_cache_len: int=0,
        kernel_initializer=None,
        bias_initializer=None,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len)

        # linear transformation for positional encoding
        self.linear_pos = keras.layers.Dense(
            n_feat,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = keras.Variable(keras.ops.zeros(( self.h, self.d_k)))
            self.pos_bias_v = keras.Variable(keras.ops.zeros(( self.h, self.d_k)))
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Example:

        T=3, the goal is to generate the relative matrix:
            [[0, -1, -2],
            [1,  0, -1]
            [2,  1,  0]].

        Steps to generate the relative matrix:
        1. create a matrix of shape (T, 2T-1)
           [[2 , 1, 0, -1, -2],
            [2 , 1, 0, -1, -2],
            [2 , 1, 0, -1, -2]]
        2. pad zero-column on the left side of the matrix with shape (T, 2T)
            [[0, 2 , 1, 0, -1, -2],
             [0, 2 , 1, 0, -1, -2],
             [0, 2 , 1, 0, -1, -2]]
        3. reshape the matrix to (2T, T)
            [[0,  2,  1],
             [0, -1, -2],
             [0,  2,  1],
             [0, -1, -2],
             [0,  2,  1],
             [0, -1, -2]]
        4. drop the first row
            [[0, -1, -2],
             [0,  2,  1],
             [0, -1, -2],
             [0,  2,  1],
             [0, -1, -2]]
        5. reshape the matrix to (T, 2T-1)
            [[0, -1, -2,  0, 2],
             [1,  0, -1, -2, 0],
             [2,  1,  0, -1, -2]]
        6. Extract the first T columns
            [[0, -1, -2],
             [1,  0, -1],
             [2,  1,  0]]
        """
        # b, h, qlen, pos_len = x.shape  # (b, h, t1, t2)
        dim = keras.ops.shape(x)
        b = dim[0]
        h = dim[1]
        qlen = dim[2]
        pos_len = dim[3]
        # need to add a column of zeros on the left side of last dimension
        # to perform the relative shifting
        keras.ops.pad
        x = tf.pad(x, paddings=[[0,0],[0,0],[0,0],[1, 0]])  # (b, h, t1, t2+1)
        x = keras.ops.reshape(x, (b, h, -1, qlen))  # (b, h, t2+1, t1)
        # need to drop the first row
        x = keras.ops.reshape(x[:, :, 1:], (b, h, qlen, pos_len)) # (b, h, t1, t2)
        return x

    def call(
            self,
            query: keras.KerasTensor,
            key: keras.KerasTensor,
            value: keras.KerasTensor,
            mask=None,
            pos_emb:keras.KerasTensor=None,
            training:bool =False)->keras.KerasTensor:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model)
                                    weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        if DEBUG:
            import numpy as np
            np.random.seed(0)
            query = np.random.randn(1,266, 176)
            query = keras.Variable(query,dtype="float32")

            key = tf.identity(query)
            value = tf.identity(query)
            print(query[0,:5,:5])

        # temporary until we solve this more gracefully
        q, k, v = self.forward_qkv(query, key, value)
        q = keras.ops.transpose(q, [0, 2, 1, 3])  # (batch, time1, head, d_k)

        n_batch_pos = keras.ops.shape(pos_emb)[0]
        p = self.linear_pos(pos_emb)
        p = keras.ops.reshape(p, (n_batch_pos, -1, self.h, self.d_k))
        p = keras.ops.transpose(p, [0, 2, 1, 3])  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = q + self.pos_bias_u
        q_with_bias_u = keras.ops.transpose(q_with_bias_u, [0, 2, 1, 3])

        # (batch, head, time1, d_k)
        q_with_bias_v = q + self.pos_bias_v
        q_with_bias_v = keras.ops.transpose(q_with_bias_v, [0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = keras.ops.matmul(
            q_with_bias_u,
            keras.ops.transpose(k, [0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = keras.ops.matmul(
            q_with_bias_v,
            keras.ops.transpose(p, [0, 1, 3, 2]))
        matrix_bd = self.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : keras.ops.shape(matrix_ac)[-1]]

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

        out = self.forward_attention(v, scores, mask, training=training)

        if DEBUG:
            print(out[0,:5,:5])

        return out

class PositionalEncoding(
    keras.layers.Layer):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(
            self,
            d_model,
            dropout_rate=0.1,
            max_len=5000,
            xscale=None,
            dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = keras.layers.Dropout(
            rate=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = keras.layers.Dropout(
                rate=dropout_rate_emb)
        else:
            self.dropout_emb = None
        self.extend_pe(max_len)

    def create_pe(self, positions):
        """Create positional encodings."""
        pos_length = keras.ops.shape(positions)[0]
        div_term = keras.ops.exp(
            tf.range(0, self.d_model, delta=2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / self.d_model)
        )
        sin= keras.ops.sin(positions * div_term)
        cos= keras.ops.cos(positions * div_term)
        pe=keras.ops.transpose(keras.layers.concatenate([sin,cos], axis=0), [1,0])
        pe=keras.ops.transpose(keras.ops.reshape(pe, (-1, pos_length)), [1,0])
        pe = keras.ops.expand_dims(pe, 0)
        self.pe = pe

    def create_pe1(self, positions):
        """Create positional encodings."""
        pos_length = keras.ops.shape(positions)[0]
        pe=np.zeros((pos_length, self.d_model))
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(np.log(10000.0) / self.d_model)
        )
        a= np.sin(positions * div_term)
        b= np.cos(positions * div_term)
        pe[:, 0::2] = a
        pe[:, 1::2] = b
        pe=tf.constant(pe, dtype=tf.float32)
        pe = tf.expand_dims(pe, 0)
        self.pe = pe

    def extend_pe(self, length):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = tf.range(0, length, dtype=tf.float32)
        positions = tf.expand_dims(positions, 1)
        self.create_pe(positions=positions)

    def forward(self, x, cache_len=0, training=False):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = keras.ops.shape(x)[0] + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb, training=training)
        x = x + pos_emb
        return self.dropout(x, training=training), pos_emb

class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """
    def __init__(
            self,
            d_model,
            dropout_rate=0.0,
            max_len=5000,
            xscale=None,
            dropout_rate_emb=0.0):
        """Construct an RelPositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            xscale=xscale,
            dropout_rate_emb=dropout_rate_emb)
        self.extend_pe(max_len)

    def extend_pe(self, length):
        """Reset and extend the positional encodings if needed."""
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = tf.range(
            length - 1,
            -length,
            delta=-1,
            dtype=tf.float32)
        positions = tf.expand_dims(positions, 1)
        self.create_pe(positions=positions)

    def call(
            self,
            x,
            cache_len=0,
            training=False):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        if self.xscale:
            x = x * self.xscale
        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = keras.ops.shape(x)[1] + cache_len
        center_pos = keras.ops.shape(self.pe)[1] // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb, training=training)
        return self.dropout(x, training=training), pos_emb
