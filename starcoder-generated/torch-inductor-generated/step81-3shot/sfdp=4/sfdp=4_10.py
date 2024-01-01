
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
Q = torch.randn(1, 128, 128, 128)
K = torch.randn(1, 128, 128, 128)
V = torch.randn(1, 128, 128, 128)
attn_mask = (torch.rand(1, 128, 128, 128) >= 0.3).float().masked_fill(torch.tensor([True, False, False, True, True, False, True, True, False, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, True, False, True]).view(1, 1, 6).expand(1, 128, 128).cuda(), float("-inf"))
