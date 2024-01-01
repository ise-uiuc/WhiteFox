
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = MultiHeadAttention(d_k)

# Inputs to the model
query = torch.randn(2, 4, 9)
key = torch.randn(2, 10, 9)
value = torch.randn(2, 10, 9)
attn_mask = torch.randn(2, 1, 4, 10)

