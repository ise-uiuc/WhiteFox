
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scaling_factor = (key_dim ** -0.5)
        self.attention = torch.nn.MultiheadAttention(embed_dim=qk_dim, num_heads=8)

    def forward(self, qk, value, is_training):
        return self.attention(qk, qk, value, need_weights=False, attn_mask=[None] * 3,
                              key_padding_mask=None,
                              dropout=0.0 if is_training else None)[0]

# Initializing the model
m = Model(64, 64, 64)

# Inputs to the model
qk = torch.randn(1, 4, 64)
value = torch.randn(1, 8, 64)
