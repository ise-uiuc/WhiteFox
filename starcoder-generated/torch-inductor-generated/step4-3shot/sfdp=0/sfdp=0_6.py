
class Model(torch.nn.Module):
    def __init__(self, num_heads = 8, hidden_key_size = 8, hidden_value_size = 8):
        super(Model, self).__init__()

        self.attn = MultiHeadAttention(num_heads, hidden_key_size, hidden_value_size)

    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None):
        return self.attn(query, key, value, attn_mask, key_padding_mask)

# Initializing the model
n = Model()

# Inputs to the model
x1 = torch.rand(1, 64, 32)
x2 = torch.rand(1, 64, 48)
x3 = torch.rand(1, 64, 32)
