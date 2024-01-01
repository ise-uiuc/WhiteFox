
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, attn_mask=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim = -1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = MultiHeadAttention(0.2)

# Inputs to the model
x1 = torch.randn(1, 32, 1024, 1024)
x2 = torch.randn(1, 32, 1024, 1024)
x3 = torch.randn(1, 32, 1024, 1024)
