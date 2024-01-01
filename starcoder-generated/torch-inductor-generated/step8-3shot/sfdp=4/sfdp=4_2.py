
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__qkv_weight__ = torch.nn.Parameter(torch.ones(16, 32, 64), requires_grad=True)
        self.__out_weight__ = torch.nn.Parameter(torch.ones(16, 32, 32), requires_grad=True)
 
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output, attn_weight

# Initializing the model
m = MultiHeadAttention()

# Inputs to the model
query = torch.randn(1, 16, 128, 32)
key = torch.randn(1, 16, 256, 32)
value = torch.randn(1, 16, 256, 32)
attn_mask = torch.randn(1, 16, 128, 256).abs() < 0.5
_, 