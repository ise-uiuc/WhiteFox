
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key_mask, value_mask, value):
        _key_mask = key_mask.unsqueeze(-2) * -1000000000
        _value_mask = value_mask.unsqueeze(-1) * -1000000000
        qk = query @ key_mask @ key_mask.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + _value_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = (attn_weight @ value + _key_mask).transpose(2, 3).transpose(1, 2)
        return output
# Inputs to the model
Q = torch.randn(1, 56, 56, 64)
key Mask = torch.randn(1, 56, 56)
value Mask = torch.randn(56, 56)
V = torch.randn(1, 64, 56, 56)
