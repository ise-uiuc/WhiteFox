
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p, max_seq_len, use_residual_bias=True):
        super(DecoderLayer, self).__init__()

        self.self_attention = SelfAttention(dim, num_heads, dropout_p, max_seq_len, use_residual_bias)
        self.pre_norm = LayerNorm(dim)
        self.attention_norm = LayerNorm(dim)

        self.ffn = FFN(dim)
        self.model_norm = LayerNorm(dim)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = self.self_attention(x)
        x = x + h # residual connection

        h = x  # pre layer norm
        x = self.attention_norm(x)
        x = self.self_ffn(x)
        x = x + h  # residual connection

        return self.model_norm(x)

# Initializing the model
m = Model(3, 4, 0.1, 100)

# Inputs to the model
x = torch.randn(8, 1, 3)
