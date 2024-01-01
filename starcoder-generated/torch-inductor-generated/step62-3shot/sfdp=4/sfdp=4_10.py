
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, queries, keys, values, attn_mask):
        qk = queries @ keys.transpose(-2, -1) / math.sqrt(queries.size(-1))
        qk = qk + attn_mask
        attention_weights = torch.softmax(qk, dim=-1)
        context = attention_weights @ values
        return context
# Inputs to the model
queries = torch.randn(1, 64, 56, 56)
keys = torch.randn(1, 64, 56, 56)
values = torch.randn(1, 64, 56, 56)
attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
