
class Model(torch.nn.Module):
    # TODO: change the hidden dimension here
    def __init__(self):
        super().__init__()
        self.heads = 32
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 16, 64)
key = torch.randn(1, 32, 16, 64)
value = torch.randn(1, 32, 16, 64)
attn_mask = torch.randn(1, 1, 16, 16)
