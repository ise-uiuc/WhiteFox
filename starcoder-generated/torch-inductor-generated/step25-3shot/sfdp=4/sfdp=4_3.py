
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Inputs to the model
torch.manual_seed(1)
q = torch.randn(2, 5, 1)
k = torch.randn(2, 1, 5)
v = torch.randn(2, 1, 5)
attn_mask = torch.tril(torch.ones((5, 5))).unsqueeze(0).unsqueeze(0).cuda()
