
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weights = torch.softmax(qk, dim=-1)
        output = attn_weights @ v
        return output, attn_weights

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 64, 64)
k = torch.randn(1, 1, 64, 64)
v = torch.randn(1, 1, 64, 64)
mask = torch.zeros(1, 1, 64, 64)
output, attn_weights = m(q, k, v, mask)
