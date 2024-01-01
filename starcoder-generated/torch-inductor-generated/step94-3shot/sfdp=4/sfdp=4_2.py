
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q0, key1, value4, attention_mask, attn_mask):
        qk = q0 @ key1.transpose(-2, -1) / math.sqrt(q0.size(-1))
        qk = qk + attention_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value4
        return output
# Inputs to the model
q0 = torch.randn(1, 64, 56, 56)
key1 = torch.randn(1, 64, 56, 56)
value4 = torch.randn(1, 64, 56, 56)
attention_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
