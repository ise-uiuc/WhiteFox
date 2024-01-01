
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, attn_mask):
        s = q.matmul(k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        s = s + attn_mask
        aw = torch.softmax(s, dim=-1)
        v1 = aw.matmul(v)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 64, 512)
k = torch.randn(1, 64, 512)
v = torch.randn(1, 64, 512)
a = 1 - torch.rand(1, 5)
attn_mask = a[:, :, None, None] * -10000.0 # this should prevent attention to positions where a[0, i] = 1
