
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query1, key2, value3, attn_mask2):
        qk = query1 @ key2.transpose(-2, -1)
        qk = qk + attn_mask2
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value3
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
