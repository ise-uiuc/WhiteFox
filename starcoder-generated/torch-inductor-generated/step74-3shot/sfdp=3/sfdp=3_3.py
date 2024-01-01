
class Model(torch.nn.Module):
    def __init__(self, model_dim, key_value_dim, num_heads, scale_factor=1. / math.sqrt(model_dim), dropout_p=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(model_dim, num_heads, dropout=dropout_p)
        self.layer_norm = torch.nn.LayerNorm(model_dim, eps=1e-12)
 
    def forward(self, query, key, value, mask=None):
        mha_output, attn_weights = self.attention(query, key, value, attention_mask=mask)
        output = self.layer_norm(query + mha_output)
        return output

# Initializing the model
m = Model(model_dim=512, key_value_dim=512, num_heads=8)

# Inputs to the model
query = torch.randn(1, 32, 512)
key = torch.randn(1, 32, 512)
value = torch.randn(1, 32, 512)
mask = (torch.ones(1, 8, 32)) == 0
