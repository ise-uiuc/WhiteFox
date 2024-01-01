
class DecoderLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, pe_dim, ff_dim, layer_norm_eps, dropout_p, scale_factor):
        super(DecoderLayer, self).__init__()

        # Layer normalization
        self.layer_norm_1 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.layer_norm_2 = LayerNorm(embed_dim, eps=layer_norm_eps)

        # Embeddings
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)
        self.embed1 = nn.Linear(pe_dim, embed_dim)
        self.embed2 = nn.Linear(pe_dim, embed_dim)

        # Self attention
        self.self_attn = MultiHeadAttention(embed_dim, n_heads, dropout_p=dropout_p)

        # Position feed forward layer
        self.pos_ffn = PoswiseFeedForwardNet(embed_dim, ff_dim)


    def forward(self, r1, r2, src_mask=None, src_attn_mask=None):
        