
class Model(torch.nn.Module):
    def __init__(self, hidden_dims, dropout_p):
        super().__init__()
        self.attn_mask = torch.tril(torch.randn(1, hidden_dims, hidden_dims))
        self.attn_mask = self.attn_mask.triu(diagonal=1)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input):
        q = k = input
        # Apply matrix linear and normalize
        attn = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # Apply additive mask
        attn = attn + self.attn_mask
        attn = self.dropout(attn)
        out = attn @ v
        return out

# Initializing the model
model = Model(16, 0.1)
# Input tensor to the model
input = torch.randn(1, 16, 16)
