import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vectot of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0) / d_model))

        # appluy the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply the cos to the uneven positino
        pe[:, 1::3] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super()._init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim= True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (batch, seq_len, d_model))
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)

        return x
    
class MultiHeadAttetionBlock(nn.Module):

    def __init__(self, d_model: int, n_heads:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model is not divisiblae by n_heads"

        self. d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model) # Wq
        self.W_k = nn.Linear(d_model, d_model) # Wk
        self.W_v = nn.Linear(d_model, d_model) # Wv

        self.W_o = nn.Linear(d_model, d_model) # Wo
        self.droppout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # ( batch, h, seq_len, seq_len )
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, Q, K, V, mask):

        query = self.W_q(Q)     # (Batch, seq_len, d_model) ---> (Batch, seq_len, d_model) 
        key = self.W_k(K)       # (Batch, seq_len, d_model) ---> (Batch, seq_len, d_model) 
        value = self.W_v(V)     # (Batch, seq_len, d_model) ---> (Batch, seq_len, d_model) 

        # (Batch, seq_len, d_model) ---> (Batch, seq_len, n_heads, d_k) ---> (Batch, n_heads, seq_len, d_k) 
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttetionBlock.attention(query=query,
                                                                    key=key,
                                                                    value=value,
                                                                    mask=mask,
                                                                    dropout=self.droppout)
        
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.W_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttetionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttetionBlock, cross_attention_block: MultiHeadAttetionBlock, feed_forward_blocl: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block,
        self.feed_forward_block = self.feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_mask(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(
    source_vocabulary_size: int,
    target_vocabulary_size: int,
    sequence_length: int,
    model_dimension: int = 512,
    num_layers: int = 6,
    num_attention_heads: int = 8,
    dropout: float = 0.1,
    feed_forward_dimension: int = 2048,
) -> Transformer:
    """Build and initialize an encoder-decoder Transformer.

    Args:
        source_vocabulary_size: Number of tokens in the source vocabulary.
        target_vocabulary_size: Number of tokens in the target vocabulary.
        sequence_length: Maximum sequence length for encoder and decoder.
        model_dimension: Size of embeddings and hidden representations.
        num_layers: Number of encoder blocks and decoder blocks.
        num_attention_heads: Number of parallel attention heads per block.
        dropout: Dropout probability used throughout the model.
        feed_forward_dimension: Hidden size of each feed-forward block.

    Returns:
        The initialized Transformer.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(model_dimension, source_vocabulary_size)
    tgt_embed = InputEmbeddings(model_dimension, target_vocabulary_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(model_dimension, sequence_length, dropout)
    tgt_pos = PositionalEncoding(model_dimension, sequence_length, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttetionBlock(
            model_dimension,
            num_attention_heads,
            dropout,
        )
        feed_forward_block = FeedForwardBlock(
            model_dimension,
            feed_forward_dimension,
            dropout,
        )
        encoder_block = EncoderBlock(
            encoder_self_attention_block,
            feed_forward_block,
            dropout,
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiHeadAttetionBlock(
            model_dimension,
            num_attention_heads,
            dropout,
        )
        decoder_cross_attention_block = MultiHeadAttetionBlock(
            model_dimension,
            num_attention_heads,
            dropout,
        )
        feed_forward_block = FeedForwardBlock(
            model_dimension,
            feed_forward_dimension,
            dropout,
        )
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(
        model_dimension,
        target_vocabulary_size,
    )

    # Create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer,
    )

    # Initialize weight matrices with Xavier uniform initialization.
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
