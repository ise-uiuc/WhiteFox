
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, num_heads, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
 
    def forward(self, src, mask=None, src_key_padding_mask=None):
        r