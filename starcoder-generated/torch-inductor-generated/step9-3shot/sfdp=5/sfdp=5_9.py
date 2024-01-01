
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The key, value, and query are initialized randomly
        self.key = torch.nn.Parameter(torch.randn(32, 16, 3, 3, 3))
        self.value = torch.nn.Parameter(torch.randn(32, 16, 128, 128, 128))
        self.query = torch.nn.Parameter(torch.randn(16, 16, 3, 3, 3))

    def forward(self, attn_mask, dropout_p):
        v1 = self.key @ self.query.transpose(-2, -1)
        v2 = v1 / math.sqrt(v1.size(-1))
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.dropout(v4, dropout_p, True)
        output = v5 @ self.value
        return output

# Initializing the model
m = Model()
dropout_p = 0.5

# Inputs to the model
attn_mask = torch.randn(64, 16, 16, 16)
