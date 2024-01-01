
class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 8)
k = torch.randn(1, 4, 8)
v = torch.randn(1, 4, 4)
