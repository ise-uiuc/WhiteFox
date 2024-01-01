
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(key.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 128, 25, 25)
key = torch.randn(1, 128, 25, 25)
value = torch.randn(1, 128, 25, 25)
attn_mask = (torch.rand(1, 128, 1, 1) > 0.7).fill_(-1000000000.0)
